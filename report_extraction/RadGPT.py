"""Lightweight LLM runner for brain CT hemorrhage extraction."""

from __future__ import annotations

import csv
import os
import re
from typing import Dict, List

import httpx
import pandas as pd
from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a knowledgeable, efficient, and direct AI assistant, and an expert in neuroradiology "
    "and brain CT radiology reports. Your goal is to transform unstructured French medical text "
    "into highly structured data for machine learning."
)

USER_PROMPT_1 = (
    "Instructions: The radiology report below describes intracranial hemorrhages "
    "(ICH, IVH, SDH, etc.) and associated findings like perihematomal edema (PHE).\n\n"
    "Read the report carefully. Your task is to extract one line per lesion or finding.\n\n"
    "Fill out the template below exactly:\n"
    "lesion X: type = _; certainty = _; size = _; structure = _; lateralization = _; "
    "ventricle_details = _; severity_phe = _; artifacts = _;\n\n"
    "Consider the following rules for brain hemorrhage:\n\n"
    "A - Types of Lesions:\n"
    "- ICH: Intracerebral/Intraparenchymal hemorrhage (Hematome/Hemorragie dans le parenchyme).\n"
    "- IVH: Intraventricular hemorrhage (Deversement/Inondation dans les ventricules).\n"
    "- PHE: Perihematomal Edema (Oedeme vasogene au pourtour).\n"
    "- SAH/SDH/EDH: Subarachnoid, Subdural, or Epidural hemorrhages.\n\n"
    "B - Specific Column Logic:\n"
    "1. size:\n"
    "   - For ICH: Always look for dimensions (e.g., 4.3 x 2.0 cm).\n"
    "   - For IVH/PHE: Use \"U\" unless a specific measurement is given.\n"
    "   - If no numbers: use 'tiny' (minime/trace), 'small' (petite), 'large' "
    "(volumineuse/importante).\n"
    "   - Use \"U\" if totally unknown.\n"
    "2. ventricle_details:\n"
    "   - For IVH: Be precise (e.g., \"corne occipitale\", \"V3\", \"V4\"). "
    "Use \"U\" if the report just says \"intraventriculaire\" without detail.\n"
    "   - For ICH: Use \"U\" UNLESS the report mentions an \"extension\" or "
    "\"deversement\" into a specific ventricle part.\n"
    "3. lateralization: Must be \"Droit\", \"Gauche\", \"Bilateral\", or \"Median\". "
    "Use \"U\" if not specified.\n"
    "4. severity_phe: Only for PHE. Use \"Leger\", \"Modere\", \"Severe/Marque\". "
    "Use \"U\" for other types.\n"
    "5. artifacts: List any mention of \"drain\", \"embolisation\", \"mouvement\", "
    "\"metal\". Otherwise, \"None\".\n\n"
    "C - Certainty & Uncertainty:\n"
    "- 'certain': Stated as a clear finding.\n"
    "- 'high': \"Probable\", \"evoquant\", \"allure de\".\n"
    "- 'low': \"Suspecte\", \"discrete trace\", \"minime foyer punctiforme\".\n"
    "- Always use \"U\" if the report is too vague to determine a type or size.\n\n"
    "D - Handling Multiple Focal Lesions:\n"
    "If a report mentions multiple distinct ICH foci (e.g., \"une frontale gauche\" "
    "and \"second foyer temporal\"), create two separate entries for ICH.\n\n"
    "E - Units:\n"
    "You MUST write the units (cm or mm) as found in the text. If not specified, assume mm.\n\n"
    "F - Language:\n"
    "The report is in French, but you must output the structured fields in English/French as "
    "specified (Type in English acroynms, locations in French).\n\n"
    "If you are sure there is no hemorrhage/edema, reply with: \"No lesions mentioned.\""
)

_CLIENT = None
_MODEL = None


def get_client(base_url: str) -> OpenAI:
    global _CLIENT, _MODEL
    if _CLIENT is not None:
        return _CLIENT

    http_client = httpx.Client(trust_env=False, verify=False)
    _CLIENT = OpenAI(api_key="dummy", base_url=base_url, http_client=http_client)
    _MODEL = _CLIENT.models.list().data[0].id
    return _CLIENT


def get_user_prompt(prompt_id: int) -> str:
    prompt = globals().get(f"USER_PROMPT_{prompt_id}")
    if prompt is None:
        raise ValueError(f"Unknown prompt_id: {prompt_id}")
    return prompt


def build_message(report: str, prompt_id: int) -> List[Dict[str, str]]:
    user_prompt = get_user_prompt(prompt_id)
    user = f"{user_prompt}\n\nReport:\n{report}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def send_message(report: str, base_url: str, prompt_id: int) -> str:
    client = get_client(base_url)
    response = client.chat.completions.create(
        model=_MODEL,
        messages=build_message(report, prompt_id),
        temperature=0,
        top_p=1,
        timeout=6000,
    )
    return response.choices[0].message.content


def parse_answer(answer: str) -> List[Dict[str, str]]:
    cleaned = answer.strip()
    if cleaned.lower().startswith("no lesions mentioned"):
        return []

    fields_template = {
        "type": "U",
        "certainty": "U",
        "size": "U",
        "structure": "U",
        "lateralization": "U",
        "ventricle_details": "U",
        "severity_phe": "U",
        "artifacts": "U",
    }
    key_aliases = {
        "ventricle details": "ventricle_details",
        "ventricle_detail": "ventricle_details",
        "severity phe": "severity_phe",
    }

    lesions = []
    pattern = re.compile(r"lesion\s*\d+\s*:\s*(.*?)(?=\n\s*lesion\s*\d+\s*:|$)", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(cleaned):
        payload = match.group(1).replace("\n", " ").strip()
        fields = dict(fields_template)

        for item in payload.split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            key = key.strip().lower().replace("_", " ")
            key = re.sub(r"\s+", " ", key)
            key = key_aliases.get(key, key).replace(" ", "_")
            value = value.strip()
            if key in fields:
                fields[key] = value

        lesions.append(fields)

    return lesions


def run_inference(
    data: pd.DataFrame,
    base_url: str,
    output_path: str,
    prompt_id: int,
    restart: bool = False,
) -> None:
    required_cols = {"ID", "Report"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    mode = "w" if restart else "a"
    write_header = restart or not os.path.exists(output_path)
    output_cols = [
        "ID",
        "Lesion Index",
        "type",
        "certainty",
        "size",
        "structure",
        "lateralization",
        "ventricle_details",
        "severity_phe",
        "artifacts",
        "DNN Answer",
        "Report",
    ]

    with open(output_path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_cols)
        if write_header:
            writer.writeheader()

        for _, row in data.iterrows():
            report = str(row["Report"])
            answer = send_message(report, base_url, prompt_id)
            print(f"ID {row['ID']} LLM answer:\n{answer}\n---")
            lesions = parse_answer(answer)

            if not lesions:
                writer.writerow(
                    {
                        "ID": row["ID"],
                        "Lesion Index": 0,
                        "type": "No lesions mentioned",
                        "certainty": "U",
                        "size": "U",
                        "structure": "U",
                        "lateralization": "U",
                        "ventricle_details": "U",
                        "severity_phe": "U",
                        "artifacts": "U",
                        "DNN Answer": answer,
                        "Report": report,
                    }
                )
                continue

            for idx, lesion in enumerate(lesions, start=1):
                writer.writerow(
                    {
                        "ID": row["ID"],
                        "Lesion Index": idx,
                        **lesion,
                        "DNN Answer": answer,
                        "Report": report,
                    }
                )