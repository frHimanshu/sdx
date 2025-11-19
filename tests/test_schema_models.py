"""Regression tests for schema modules previously lacking coverage."""

from __future__ import annotations

from datetime import datetime, timezone

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
    """Patient inherits the BaseLanguage mixin and preserves language values."""

    patient = Patient.model_construct(language='es-MX')
    assert patient.language == 'es-MX'


def test_encounter_canonical_episode_and_language() -> None:
    """Encounter adds canonicalEpisodeId while keeping BaseLanguage behavior."""

    encounter = Encounter.model_construct(
        language='en-US',
        canonicalEpisodeId='episode-123',
    )

    assert encounter.language == 'en-US'
    assert encounter.canonicalEpisodeId == 'episode-123'


def test_other_fhir_resources_share_language_field() -> None:
    """Sanity-check remaining subclasses that only add BaseLanguage."""

    observation = Observation.model_construct(language='fr-FR')
    condition = Condition.model_construct(language='pt-BR')
    procedure = Procedure.model_construct(language='de-DE')
    clinical_impression = ClinicalImpression.model_construct(language='it-IT')
    annotation = Annotation.model_construct(language='ja-JP')

    assert observation.language == 'fr-FR'
    assert condition.language == 'pt-BR'
    assert procedure.language == 'de-DE'
    assert clinical_impression.language == 'it-IT'
    assert annotation.language == 'ja-JP'


def test_human_evaluations_models_support_language() -> None:
    """Instances from human_evaluations set and expose the language field."""

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

