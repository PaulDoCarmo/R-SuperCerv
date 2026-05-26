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

USER_PROMPT_2 = (
    "Instructions: The radiology report below describes intracranial hemorrhages "
    "(ICH, IVH, SDH, etc.) and associated findings like perihematomal edema (PHE).\n\n"
    "Read the report carefully. Every finding must be isolated.\n"
    "Fill out the template below exactly:\n"
    "lesion X: type = _; certainty = _; size = _; structure = _; lateralization = _; "
    "artifacts = _;\n\n"
    "Consider the following rules for brain hemorrhage:\n"
    "1. MAPPING & DEFINITIONS (Types)\n"
    "ICH: Intracerebral / Intraparenchymal Hemorrhage (Hematome/Hemorragie intraparenchymateuse).\n"
    "IVH: Intraventricular Hemorrhage (Deversement/Inondation intraventriculaire).\n"
    "PHE: Perihematomal Edema (Oedeme vasogene au pourtour).\n"
    "SAH: Subarachnoid Hemorrhage (Hemorragie sous-arachnoidienne).\n"
    "SDH: Subdural Hemorrhage (Hematome sous-dural).\n"
    "EDH: Epidural Hemorrhage (Hematome extra-dural).\n"
    "2. EXTRACTION RULES\n"
    "ATOMICITY (The \"No Conjunction\" Rule): Generate one line per structure. If a lesion is "
    "\"Frontal et Parietal\", you MUST generate two lines: one for Frontal and one for Parietal. "
    "If a report mentions multiple distinct ICH foci (e.g., \"une frontale gauche\" and "
    "\"second foyer temporal\"), create two separate entries for ICH.\n"
    "STRUCTURE vs LATERALIZATION: The structure column must ONLY contain the anatomical name "
    "(e.g., \"Frontal\", \"Thalamus\", \"V3\"). The side (Left, Right) MUST be moved to the "
    "lateralization column.\n"
    "Example: \"Lenticulaire gauche\" -> structure: Lenticular; lateralization: Left.\n"
    "COLUMN MAPPING:\n"
    "size: Use for dimensions (e.g., \"2x3 cm\") or descriptors (\"Small\", \"Large\"). "
    "You MUST write the units (cm or mm) as found in the text. If not specified, assume mm. "
    "For PHE, put the severity here (\"Mild\", \"Moderate\", \"Severe\").\n"
    "structure :\n"
    "- For ICH/SAH/SDH/EDH: The brain lobe or region.\n"
    "- For IVH: The specific ventricle name (e.g., V3, V4, Corne occipitale).\n"
    "- For PHE: The region it surrounds.\n"
    "lateralization: Must be \"Left\", \"Right\", \"Bilateral\", \"Median\", or \"U\".\n"
    "HALLUCINATION PREVENTION: For every column EXCEPT type, if the information is not explicitly "
    "mentioned or is unclear, you MUST use \"U\".\n"
    "LANGUAGE: The output must be 100% in English. Translate anatomical terms (e.g., "
    "\"tronc cerebral\" -> \"Brainstem\").\n"
    "3. LOGIC VALUES\n"
    "CERTAINTY: certain, high (probable), low (suspected/trace).\n"
    "ARTIFACTS: drain, embolization, movement, metal, None.\n"
    "If you are sure there is no hemorrhage/edema, reply with: \"No lesions mentioned.\""
)

USER_PROMPT_3 = (
    "Instructions: The radiology report below describes intracranial hemorrhages "
    "(ICH, IVH, SDH, etc.) and associated findings like perihematomal edema (PHE).\n\n"
    "Read the report carefully. Your task is to list the types, certainty of lesion type, "
    "sizes, locations (structure and lateralisation) of all lesions/findings in the report.\n\n"
    "Fill out the template below, using one line per lesion (you may add or remove lines from "
    "the template):\n"
    "   lesion 1: type = _; certainty = _; size = _; structure = _; lateralization = _;\n"
    "   lesion 2:  type = _; certainty = _; size = _; structure = _; lateralization = _;\n\n"
    "If you are absolutely sure the report mentions no lesion, do not use the template. "
    "Instead, reply with: 'No lesions mentioned.' and justify why you are sure the report "
    "mentions no lesion.\n\n"
    "The report is in French but the output must be 100% in English. Translate anatomical terms "
    "(e.g., \"tronc cerebral\" -> \"Brainstem\").\n\n"
    "Consider the following instructions :\n\n"
    "A - Types of Lesions:\n"
    "- ICH: Intracerebral/Intraparenchymal hemorrhage (Hematome/Hemorragie dans le parenchyme).\n"
    "- IVH: Intraventricular hemorrhage (Deversement/Inondation dans les ventricules).\n"
    "- PHE: Perihematomal Edema (Oedeme vasogene au pourtour).\n"
    "- SAH/SDH/EDH: Subarachnoid, Subdural, or Epidural hemorrhages.\n\n"
    "B - Specific Column Logic:\n"
    "1. size:\n"
    "   - For ICH: Always look for dimensions (e.g., 4.3 x 2.0 cm).\n"
    "   - For IVH: Use \"U\" unless a specific measurement or any qualitative indication is "
    "given.\n"
    "   - For PHE : Put the severity (\"Mild\", \"Moderate\", \"Severe\").\n"
    "   - You MUST write the units (cm or mm) as found in the text. If not specified, assume mm.\n"
    "   - If no numbers: use 'tiny' (minime/trace), 'small' (petite), 'large' "
    "(volumineuse/importante).\n"
    "   - Use \"U\" if totally unknown.\n\n"
    "2. structure :\n"
    "- For ICH/SAH/SDH/EDH: The brain lobe or region. Use standard names.\n"
    "- For IVH: The specific ventricle name (e.g., V3, V4, Corne occipitale) if provided\n"
    "- For PHE: The region it surrounds.\n"
    "- Use \"U\" if unknown.\n"
    "Extraction rules for structures :\n"
    "- The structure column must ONLY contain the anatomical name (e.g., \"Frontal\", "
    "\"Thalamus\", \"V3\"). The side (Left, Right) MUST be moved to the lateralization column.\n"
    "Example: \"Lenticulaire gauche\" -> structure: Lenticular; lateralization: Left.\n"
    "- Generate one line per structure. If a lesion/finding is \"Frontal et Parietal\", you MUST "
    "generate two lines: one for Frontal and one for Parietal. If a report mentions multiple "
    "distinct ICH foci (e.g., \"une frontale gauche\" and \"second foyer temporal\"), create two "
    "separate entries for ICH.\n"
    "3. Lateralization: Must be \"Left\", \"Right\", \"Bilateral\", \"Median\", or \"U\" if unknown.\n"
    "4. Certainty :\n"
    "Certainty of the lesion type, according to the report. If a report mentions a lesion type "
    "in the findings, history, or impressions, without demonstrating uncertainty, say certainty "
    "= certain.\n"
    "If the report expresses strong confidence in lesion type, say certainty = high.\n"
    "If the report mentions a lesion type but expresses significant uncertainty about it, say "
    "certainty = low.\n"
    "C - Justification:\n"
    "Besides filling the template, justify your answer, carefully mentioning each section of "
    "the report if present: history, findings, and impressions.\n"
    "Explain from which sentences you got each size, location, and type."
)

USER_PROMPT_4 = (
    "Instructions: The radiology report below describes intracranial hemorrhages "
    "(ICH, IVH, SDH, etc.) and associated findings like perihematomal edema (PHE).\n\n"
    "Read the report carefully. Your task is to list the types, certainty of lesion type, "
    "sizes, locations (structure and lateralisation) of all lesions/findings in the report.\n\n"
    "Fill out the template below, using one line per lesion (you may add or remove lines from "
    "the template):\n"
    "   lesion 1: type = _; certainty = _; size = _; structure = _; lateralization = _;\n"
    "   lesion 2:  type = _; certainty = _; size = _; structure = _; lateralization = _;\n\n"
    "If you are absolutely sure the report mentions no lesion, do not use the template. "
    "Instead, reply with: 'No lesions mentioned.' and justify why you are sure the report "
    "mentions no lesion.\n\n"
    "The report is in french but the output must be 100% in English. Translate anatomical terms "
    "(e.g., \"tronc cerebral\" -> \"Brainstem\").\n\n"
    "Consider the following instructions :\n\n"
    "A - Types of Lesions:\n"
    "- ICH: Intracerebral/Intraparenchymal hemorrhage (Hematome/Hemorragie dans le parenchyme).\n"
    "- IVH: Intraventricular hemorrhage (Deversement/Inondation dans les ventricules).\n"
    "- PHE: Perihematomal Edema (In French reports, look for \"oedeme vasogene au pourtour\", "
    "but also implicit terms like \"envahissement au pourtour\", \"infiltration peripherique\" ).\n"
    "- SAH : Subarachnoid hemorrhages (Hemorragies sous-arachnoidiennes)\n"
    "- SDH: Subdural hemorrhages (Hemorragies sous-durales)\n"
    "- EDH: Epidural hemorrhages (Hemorragies epidurales)\n\n"
    "B - Specific Column Logic:\n"
    "1. size: \n"
    "   - For ICH: Always look for dimensions (e.g., 4.3 x 2.0 cm). \n"
    "   - For IVH: Use \"U\" unless a specific measurement or any qualitative indication is given. \n"
    "   - For PHE : Put the severity (\"Mild\", \"Moderate\", \"Severe\"). The severity must "
    "strictly be one of these three values: \"Mild\", \"Moderate\", or \"Severe\", or \"U\" "
    "(if severity is completely unknown or not specified). Map terms like \"important\", "
    "\"majeur\", \"volumineux\", \"extensif\", or \"marque\" to \"Severe\". "
    "Map \"modere\" to \"Moderate\". Map \"discret\", \"leger\", or \"minime\" to \"Mild\". \n"
    "   - You MUST write the units (cm or mm) as found in the text. If not specified, assume mm.\n"
    "   - If no numbers: use 'tiny' (minime/trace), 'small' (petite), 'large' "
    "(volumineuse/importante).\n"
    "   - Use \"U\" if totally unknown.\n\n"
    "2. structure : \n"
    "- For ICH/SAH/SDH/EDH: The brain lobe or region. Use standard names.\n"
    "- For IVH: The specific ventricle name (e.g., Lateral Ventricle, V3, V4, Corne "
    "occipitale) if provided. If the report uses global or non-specific terms like \"ensemble "
    "des cavites ventriculaires\", \"inondation ventriculaire totale/comlete\", "
    "\"hemorragie intraventriculaire multifocale\", or simply \"importante hemorragie "
    "intraventriculaire\" without naming specific ventricles, you MUST assume all 4 major "
    "cavities are affected. To respect the \"one line per finding\" rule, group this global "
    "event into one single entry using slashes (/) to separate the compartments: \n"
    "structure = Lateral Ventricle/V3/V4\n"
    "lateralization = Bilateral/Median/Median\n"
    "- For PHE: The region it surrounds. \n"
    "- Use \"U\" if unknown.\n"
    "Extraction rules for structures : \n"
    "- The structure column must ONLY contain the anatomical name (e.g., \"Frontal\", "
    "\"Thalamus\", \"V3\"). The side (Left, Right) MUST be moved to the lateralization column.\n"
    "Example: \"Lenticulaire gauche\" -> structure: Lenticular; lateralization: Left.\n"
    "-One line per distinct lesion focus: Do NOT split a single hematoma into multiple lines "
    "just because it spans across two or more regions (e.g., \"Parieto-occipital\" describes "
    "one single lesion at the border of two lobes, so it must remain on one single line with "
    "structure = Parieto-occipital. Other example : if a lesion is parietal and frontal : "
    "structure = ). \n"
    "-Multiple foci: You MUST generate separate lines only when the report describes anatomically "
    "distinct and separate hemorrhagic foci (e.g., \"une parieto-occipitale de 3.4 x 2.5 cm\" "
    "AND a \"second foyer temporal gauche de 2.8 x 1.9 cm\" must generate exactly two separate "
    "entries for ICH, because they are two distinct hematomas).\n"
    "3. Lateralization: \n"
    "Must be \"Left\", \"Right\", \"Bilateral\", \"Median\", or \"U\" if unknown.\n"
    "For IVH : Lateral structures (e.g., \"Lateral Ventricle\", \"Occipital Horn\", "
    "\"Frontal Horn\") CAN NEVER have \"Median\" as their lateralization. If the text "
    "describes an ambiguous midline pooling that affects these lateral structures, use "
    "\"Bilateral\". The lateralization \"Median\" is strictly and exclusively reserved for "
    "inherently central structures like \"V3\" and \"V4\". \n"
    "4. Certainty : \n"
    "Certainty of the lesion type, according to the report. If a report mentions a lesion type "
    "in the findings, history, or impressions, without demonstrating uncertainty, say certainty "
    "= certain. \n"
    "If the report expresses strong confidence in lesion type, say certainty = high.\n"
    "If the report mentions a lesion type but expresses significant uncertainty about it, say "
    "certainty = low. \n"
    "C - Justification:\n"
    "Besides filling the template, justify your answer, carefully mentioning each section of "
    "the report if present: history, findings, and impressions.\n"
    "Detailed Sentence Mapping: Explain from which specific sentences you extracted each lesion "
    "type, certainty, size, structure, and lateralization.\n"
    "Justification of Absence: If any of the core findings (ICH, IVH, or PHE) are NOT detected, "
    "you MUST explicitly justify their absence in your explanation (e.g., \"No ICH was detected "
    "because the report contains no mention of intraparenchymal hematoma or bleeding within the "
    "brain tissue\"). Every category must be accounted for in the justification."
)

USER_PROMPT_5 = (
    "Instructions: The radiology report below describes intracranial hemorrhages "
    "(ICH, IVH, SDH, etc.) and associated findings like perihematomal edema (PHE).\n\n"
    "Read the report carefully. Your task is to list the types, certainty of lesion type, "
    "sizes, locations (structure and lateralisation) of all lesions/findings in the report.\n\n"
    "Fill out the template below, using one line per lesion (you may add or remove lines from "
    "the template):\n"
    "   lesion 1: type = _; certainty = _; size = _; structure = _; lateralization = _;\n"
    "   lesion 2:  type = _; certainty = _; size = _; structure = _; lateralization = _;\n\n"
    "If you are absolutely sure the report mentions no lesion, do not use the template. "
    "Instead, reply with: 'No lesions mentioned.' and justify why you are sure the report "
    "mentions no lesion.\n\n"
    "The report is in french but the output must be 100% in English. Translate anatomical terms "
    "(e.g., \"tronc cerebral\" -> \"Brainstem\").\n\n"
    "Consider the following instructions :\n\n"
    "A - Types of Lesions:\n"
    "- ICH: Intracerebral/Intraparenchymal hemorrhage (Hematome/Hemorragie dans le parenchyme).\n"
    "- IVH: Intraventricular hemorrhage (Deversement/Inondation dans les ventricules).\n"
    "- PHE: Perihematomal Edema (In French reports, look for \"oedeme vasogene au pourtour\", "
    "but also implicit terms like \"envahissement au pourtour\", \"infiltration peripherique\" ).\n"
    "- SAH : Subarachnoid hemorrhages (Hemorragies sous-arachnoidiennes)\n"
    "- SDH: Subdural hemorrhages (Hemorragies sous-durales)\n"
    "- EDH: Epidural hemorrhages (Hemorragies epidurales)\n\n"
    "B - Specific Column Logic:\n"
    "1. size: \n"
    "   - For ICH: Always look for dimensions (e.g., 4.3 x 2.0 cm). \n"
    "   - For IVH: Use \"U\" unless a specific measurement or any qualitative indication is given. \n"
    "   - For PHE : Put the severity (\"Mild\", \"Moderate\", \"Severe\"). The severity must "
    "strictly be one of these three values: \"Mild\", \"Moderate\", or \"Severe\", or \"U\" "
    "(if severity is completely unknown or not specified). Map terms like \"important\", "
    "\"majeur\", \"volumineux\", \"extensif\", or \"marque\" to \"Severe\". "
    "Map \"modere\" to \"Moderate\". Map \"discret\", \"leger\", or \"minime\" to \"Mild\". \n"
    "   - You MUST write the units (cm or mm) as found in the text. If not specified, assume mm.\n"
    "   - If no numbers: use 'tiny' (minime/trace), 'small' (petite), 'large' "
    "(volumineuse/importante).\n"
    "   - Use \"U\" if totally unknown.\n\n"
    "2. structure : \n"
    "- For ICH/SAH/SDH/EDH: The brain lobe or region. Use standard names.\n"
    "- For IVH: The specific ventricle name (e.g., Lateral Ventricle, V3, V4, Corne "
    "occipitale) if provided. If the report uses global or non-specific terms like \"ensemble "
    "des cavites ventriculaires\", \"inondation ventriculaire totale/comlete\", "
    "\"hemorragie intraventriculaire multifocale\", or simply \"importante hemorragie "
    "intraventriculaire\" without naming specific ventricles, you MUST assume all 4 major "
    "cavities are affected. To respect the \"one line per finding\" rule, group this global "
    "event into one single entry using slashes (/) to separate the compartments: \n"
    "structure = Lateral Ventricle/V3/V4\n"
    "lateralization = Bilateral/Median/Median\n"
    "- For PHE: The region it surrounds. \n"
    "- Use \"U\" if unknown.\n"
    "Extraction rules for structures : \n"
    "- The structure column must ONLY contain the anatomical name (e.g., \"Frontal\", "
    "\"Thalamus\", \"V3\"). The side (Left, Right) MUST be moved to the lateralization column.\n"
    "Example: \"Lenticulaire gauche\" -> structure: Lenticular; lateralization: Left.\n"
    "-One line per distinct lesion focus: Do NOT split a single hematoma into multiple lines "
    "just because it spans across two or more regions (e.g., \"Parieto-occipital\" describes "
    "one single lesion at the border of two lobes, so it must remain on one single line with "
    "structure = Parieto-occipital. Other example : if a lesion is parietal and frontal : "
    "structure = ). \n"
    "-Multiple foci: You MUST generate separate lines only when the report describes anatomically "
    "distinct and separate hemorrhagic foci (e.g., \"une parieto-occipitale de 3.4 x 2.5 cm\" "
    "AND a \"second foyer temporal gauche de 2.8 x 1.9 cm\" must generate exactly two separate "
    "entries for ICH, because they are two distinct hematomas).\n"
    "3. Lateralization: \n"
    "Must be \"Left\", \"Right\", \"Bilateral\", \"Median\", or \"U\" if unknown.\n"
    "For IVH : Lateral structures (e.g., \"Lateral Ventricle\", \"Occipital Horn\", "
    "\"Frontal Horn\") CAN NEVER have \"Median\" as their lateralization. If the text "
    "describes an ambiguous midline pooling that affects these lateral structures, use "
    "\"Bilateral\". The lateralization \"Median\" is strictly and exclusively reserved for "
    "inherently central structures like \"V3\" and \"V4\". \n"
    "Slash Alignment Rule: The number of elements separated by slashes (/) in the structure "
    "column MUST EXACTLY MATCH the number of elements in the lateralization column."
    "If you write 2 structures, you MUST write 2 lateralizations (e.g., structure = "
    "Parieto-occipital/Cerebellum -> lateralization = Left/Left or Left/Right). "
    "Never write a single word like Left if there are multiple structures."
    "For the Global IVH exception, you MUST strictly follow the example and write 3 elements "
    "for both columns: structure = Lateral Ventricle/V3/V4 -> lateralization = "
    "Bilateral/Median/Median. Count the slashes; they must match perfectly.\n"
    "4. Certainty : \n"
    "Certainty of the lesion type, according to the report. If a report mentions a lesion type "
    "in the findings, history, or impressions, without demonstrating uncertainty, say certainty "
    "= certain. \n"
    "If the report expresses strong confidence in lesion type, say certainty = high.\n"
    "If the report mentions a lesion type but expresses significant uncertainty about it, say "
    "certainty = low. \n"
    "C - Justification:\n"
    "Besides filling the template, justify your answer, carefully mentioning each section of "
    "the report if present: history, findings, and impressions.\n"
    "Detailed Sentence Mapping: Explain from which specific sentences you extracted each lesion "
    "type, certainty, size, structure, and lateralization.\n"
    "Justification of Absence: If any of the core findings (ICH, IVH, or PHE) are NOT detected, "
    "you MUST explicitly justify their absence in your explanation (e.g., \"No ICH was detected "
    "because the report contains no mention of intraparenchymal hematoma or bleeding within the "
    "brain tissue\"). Every category must be accounted for in the justification"
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


def get_prompt_schema(prompt_id: int) -> Dict[str, object]:
    base_fields = [
        "type",
        "certainty",
        "size",
        "structure",
        "lateralization",
    ]
    prompt_fields = {
        1: [
            *base_fields,
            "ventricle_details",
            "severity_phe",
            "artifacts",
        ],
        2: [
            *base_fields,
            "artifacts",
        ],
        3: base_fields,
        4: base_fields,
        5: base_fields,
    }
    key_aliases_by_prompt = {
        1: {
            "ventricle details": "ventricle_details",
            "ventricle_detail": "ventricle_details",
            "severity phe": "severity_phe",
        },
    }

    output_fields = prompt_fields.get(prompt_id)
    if output_fields is None:
        raise ValueError(f"Unknown prompt_id: {prompt_id}")

    key_aliases = key_aliases_by_prompt.get(prompt_id, {})

    fields_template = {field: "U" for field in output_fields}
    return {
        "output_fields": output_fields,
        "fields_template": fields_template,
        "key_aliases": key_aliases,
    }


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


def parse_answer(answer: str, prompt_id: int) -> List[Dict[str, str]]:
    cleaned = answer.strip()
    if cleaned.lower().startswith("no lesions mentioned"):
        return []

    schema = get_prompt_schema(prompt_id)
    fields_template = schema["fields_template"]
    key_aliases = schema["key_aliases"]

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
    schema = get_prompt_schema(prompt_id)
    output_cols = [
        "ID",
        "Lesion Index",
        *schema["output_fields"],
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
            lesions = parse_answer(answer, prompt_id)

            if not lesions:
                empty_row = {field: "U" for field in schema["output_fields"]}
                empty_row["type"] = "No lesions mentioned"
                writer.writerow(
                    {
                        "ID": row["ID"],
                        "Lesion Index": 0,
                        **empty_row,
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