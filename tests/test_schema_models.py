"""Regression tests for schema modules previously lacking coverage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pydantic import ValidationError
from sdx.schema.fhirx import (
    Annotation,
    ClinicalImpression,
    Condition,
    Encounter,
    Observation,
    Patient,
    Procedure,
)
from sdx.schema.human_evaluations import (
    AIOutput,
    DeIdentifiedDatasetDescriptor,
    Evaluation,
)


def test_patient_language_field_round_trip() -> None:
    """Patient instances keep the BaseLanguage tag intact."""
    patient = Patient.model_validate({'language': 'es-MX'})
    assert patient.language == 'es-MX'


def test_encounter_canonical_episode_and_language() -> None:
    """Encounter stores canonicalEpisodeId and BaseLanguage details."""
    encounter = Encounter.model_validate(
        {'language': 'en-US', 'canonicalEpisodeId': 'episode-123'}
    )

    assert encounter.language == 'en-US'
    assert encounter.canonicalEpisodeId == 'episode-123'


def test_other_fhir_resources_share_language_field() -> None:
    """Remaining subclasses also expose the common language field."""
    observation = Observation.model_validate({'language': 'fr-FR'})
    condition = Condition.model_validate({'language': 'pt-BR'})
    procedure = Procedure.model_validate({'language': 'de-DE'})
    clinical_impression = ClinicalImpression.model_validate(
        {'language': 'it-IT'}
    )
    annotation = Annotation.model_validate({'language': 'ja-JP'})

    assert observation.language == 'fr-FR'
    assert condition.language == 'pt-BR'
    assert procedure.language == 'de-DE'
    assert clinical_impression.language == 'it-IT'
    assert annotation.language == 'ja-JP'


def test_human_evaluations_models_support_language() -> None:
    """Human evaluation models surface the optional language attribute."""
    timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ai_output = AIOutput(
        id='out-1',
        encounter_id='enc-9',
        type='diagnosis',
        content='Diagnosis text',
        model_version='gpt-v1',
        timestamp=timestamp,
        language='en-US',
    )
    evaluation = Evaluation(
        id='eval-1',
        aioutput_id=ai_output.id,
        output_type=ai_output.type,
        ratings={
            'accuracy': 5,
            'relevance': 4,
            'usefulness': 4,
            'coherence': 5,
        },
        safety='safe',
        comments='Looks good',
        timestamp=timestamp,
        language='en-US',
    )
    dataset_descriptor = DeIdentifiedDatasetDescriptor(
        dataset_id='ds-1',
        generation_date=timestamp,
        version='2024.01',
        records=12,
        license='CC-BY-4.0',
        url='https://example.com/datasets/ds-1',
        language='en-US',
    )

    assert ai_output.language == 'en-US'
    assert evaluation.language == 'en-US'
    assert evaluation.ratings['accuracy'] == 5
    assert dataset_descriptor.language == 'en-US'
    assert dataset_descriptor.url.endswith('ds-1')


def test_patient_language_validation() -> None:
    """Patient.language enforces BaseLanguage validation."""
    with pytest.raises(ValidationError):
        Patient.model_validate({'language': 'not-a-valid-tag'})
