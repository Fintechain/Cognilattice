# -*- coding: utf-8 -*-
"""Profile-aware memory scoring.

The profile layer gives Memery a durable attention policy and a professional
growth direction. Traits decide what becomes salient; the professional profile
decides which objects, methods, evaluation criteria, and error patterns matter.
"""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any


TRAIT_KEYS = (
    "exploration",
    "rigor",
    "conscientiousness",
    "risk_sensitivity",
    "social_warmth",
    "abstraction_preference",
    "execution",
)

DEFAULT_CALIBRATION = {
    "trait_weight": 0.12,
    "profession_weight": 0.18,
    "learning_weight": 0.10,
    "risk_penalty_weight": 0.08,
    "store_threshold": 0.75,
    "retrieve_threshold": 0.35,
}

MEMORY_IDENTITY_SKELETON = {
    "version": "memory_identity_skeleton_v1",
    "purpose": (
        "Represent long-term AI memory as a composed cognitive scaffold rather "
        "than a flat transcript archive."
    ),
    "source_basis": [
        {
            "name": "O*NET Content Model",
            "url": "https://www.onetcenter.org/content.html",
            "use": "Organize occupation memory around worker characteristics, requirements, experience, occupational requirements, workforce characteristics, and occupation-specific information.",
        },
        {
            "name": "BLS Standard Occupational Classification",
            "url": "https://www.bls.gov/soc/",
            "use": "Keep profession presets compatible with major/minor/broad/detailed occupation thinking.",
        },
        {
            "name": "ESCO",
            "url": "https://esco.ec.europa.eu/en/about-esco/what-esco",
            "use": "Model occupations together with skills, competences, knowledge, and their relationships.",
        },
    ],
    "layers": [
        "personality_attention",
        "trait_weights",
        "profession_identity",
        "professional_objects",
        "professional_methods",
        "evaluation_rubric",
        "case_library",
        "error_library",
        "retrieval_policy",
        "consolidation_policy",
    ],
    "prompt": {
        "identity": [
            "Memory has an identity scaffold: personality decides salience and pressure; profession decides objects, methods, evaluation, cases, and errors.",
            "A memory is valuable when it changes future attention, action, judgment, or professional error avoidance.",
        ],
        "write_policy": [
            "Do not store everything. Store what fits the active personality-profession tendency or what corrects it.",
            "Separate raw episode, stable semantic claim, reusable procedure, and error pattern.",
        ],
        "retrieval_policy": [
            "Retrieve by task relevance, professional cue overlap, evidence quality, and error-prevention value.",
            "When the task is risky or uncertain, retrieve error memories earlier.",
        ],
        "consolidation_policy": [
            "Promote repeated episodes into professional patterns only with enough shared cues.",
            "Promote repeated success into procedural templates and repeated failure into prevention rules.",
        ],
    },
}

DEFAULT_PROFILE_PACKS: dict[str, dict[str, Any]] = {
    "generalist_v1": {
        "profile_id": "generalist_v1",
        "name": "Generalist",
        "description": "Balanced long-term memory for broad personal and project contexts.",
        "version": "1",
        "traits": {
            "exploration": 0.55,
            "rigor": 0.65,
            "conscientiousness": 0.65,
            "risk_sensitivity": 0.55,
            "social_warmth": 0.55,
            "abstraction_preference": 0.55,
            "execution": 0.55,
        },
        "profession": {
            "role": "generalist",
            "objects": ["goal", "constraint", "preference", "event", "decision", "risk"],
            "methods": ["clarify", "summarize", "prioritize", "plan", "review"],
            "evaluation": ["usefulness", "clarity", "consistency", "timeliness"],
            "case_library_refs": [],
            "error_library_refs": [],
        },
        "development_plan": {
            "target_role": "generalist",
            "stage": "foundation",
            "focus_areas": ["goals", "preferences", "decisions", "risks"],
            "promotion_rules": [
                "Promote repeated episodes into stable preferences or principles.",
                "Promote repeated failures into error patterns and avoidance rules.",
            ],
        },
        "calibration": DEFAULT_CALIBRATION,
    },
    "software_engineer_v1": {
        "profile_id": "software_engineer_v1",
        "name": "Software Engineer",
        "description": "Memory growth toward requirements, architecture, interfaces, tests, and failure modes.",
        "version": "1",
        "traits": {
            "exploration": 0.55,
            "rigor": 0.82,
            "conscientiousness": 0.84,
            "risk_sensitivity": 0.78,
            "social_warmth": 0.45,
            "abstraction_preference": 0.86,
            "execution": 0.82,
        },
        "profession": {
            "role": "software_engineer",
            "objects": [
                "requirement", "constraint", "interface", "architecture",
                "tradeoff", "failure_mode", "test", "deployment", "dependency",
            ],
            "methods": [
                "clarify_requirements", "decompose_system", "compare_alternatives",
                "implement", "debug", "review", "refactor", "verify", "validate",
            ],
            "evaluation": [
                "correctness", "maintainability", "reliability", "security",
                "performance", "simplicity", "test_coverage",
            ],
            "case_library_refs": ["swebench", "project_postmortems"],
            "error_library_refs": ["regression_failures", "incident_postmortems"],
        },
        "development_plan": {
            "target_role": "software_engineer",
            "stage": "foundation",
            "focus_areas": [
                "requirements", "interfaces", "architecture", "tests",
                "failure_modes", "tradeoffs",
            ],
            "promotion_rules": [
                "Promote repeated implementation episodes into design patterns.",
                "Promote repeated failures into regression checks and review rules.",
                "Promote stable interface facts into semantic memory.",
            ],
        },
        "calibration": DEFAULT_CALIBRATION,
    },
    "research_scientist_v1": {
        "profile_id": "research_scientist_v1",
        "name": "Research Scientist",
        "description": "Memory growth toward hypotheses, evidence, methods, confounders, and replication.",
        "version": "1",
        "traits": {
            "exploration": 0.88,
            "rigor": 0.93,
            "conscientiousness": 0.78,
            "risk_sensitivity": 0.62,
            "social_warmth": 0.35,
            "abstraction_preference": 0.82,
            "execution": 0.58,
        },
        "profession": {
            "role": "research_scientist",
            "objects": [
                "question", "hypothesis", "variable", "method", "evidence",
                "confounder", "replication", "uncertainty", "alternative_explanation",
            ],
            "methods": [
                "formulate", "operationalize", "experiment", "analyze",
                "replicate", "compare_explanations", "revise_theory",
            ],
            "evaluation": [
                "validity", "reproducibility", "effect_size",
                "explanatory_power", "novelty", "falsifiability",
            ],
            "case_library_refs": ["paper_reading_corpus", "experiment_logs"],
            "error_library_refs": ["failed_replications", "retraction_screening"],
        },
        "development_plan": {
            "target_role": "research_scientist",
            "stage": "foundation",
            "focus_areas": [
                "hypotheses", "variables", "evidence", "confounders",
                "replication", "uncertainty",
            ],
            "promotion_rules": [
                "Promote repeated observations into hypotheses or known confounders.",
                "Promote failed experiments into methodological cautions.",
                "Promote replicated findings into stable semantic memory.",
            ],
        },
        "calibration": DEFAULT_CALIBRATION,
    },
    "clinical_reasoner_v1": {
        "profile_id": "clinical_reasoner_v1",
        "name": "Clinical Reasoner",
        "description": "Memory growth toward differentials, red flags, risk stratification, and safety checks.",
        "version": "1",
        "traits": {
            "exploration": 0.42,
            "rigor": 0.94,
            "conscientiousness": 0.90,
            "risk_sensitivity": 0.97,
            "social_warmth": 0.78,
            "abstraction_preference": 0.78,
            "execution": 0.72,
        },
        "profession": {
            "role": "clinical_reasoner",
            "objects": [
                "symptom", "history", "sign", "risk_factor",
                "differential_diagnosis", "test", "red_flag",
                "contraindication", "follow_up",
            ],
            "methods": [
                "collect_history", "triage", "differential",
                "risk_stratify", "test_selection", "safety_net", "follow_up",
            ],
            "evaluation": [
                "patient_safety", "diagnostic_coverage", "guideline_alignment",
                "false_negative_risk", "clarity", "abstention_quality",
            ],
            "case_library_refs": ["clinical_cases"],
            "error_library_refs": ["diagnostic_errors", "near_miss_cases"],
        },
        "development_plan": {
            "target_role": "clinical_reasoner",
            "stage": "supervised_foundation",
            "focus_areas": [
                "red_flags", "differentials", "risk_factors",
                "contraindications", "follow_up",
            ],
            "promotion_rules": [
                "Promote repeated cases into structured illness-script-like patterns.",
                "Promote unsafe or uncertain cases into red-flag and escalation rules.",
                "Keep high-risk conclusions conservative and evidence-gated.",
            ],
        },
        "calibration": DEFAULT_CALIBRATION,
    },
}

PERSONALITY_PRESETS: dict[str, dict[str, Any]] = {
    "balanced_operator_v1": {
        "personality_id": "balanced_operator_v1",
        "name": "Balanced Operator",
        "description": "Keeps breadth, rigor, action, and risk in stable balance.",
        "traits": {
            "exploration": 0.58, "rigor": 0.66, "conscientiousness": 0.66,
            "risk_sensitivity": 0.58, "social_warmth": 0.56,
            "abstraction_preference": 0.60, "execution": 0.62,
        },
        "attention_prompt": [
            "Keep goals, constraints, evidence, risks, preferences, and next actions in balance.",
            "Prefer memories that remain useful across more than one future task.",
        ],
        "execution_prompt": [
            "Choose reliable progress over clever detours.",
            "Convert uncertainty into small checks and clear follow-up actions.",
        ],
        "memory_prompt": [
            "Preserve stable decisions, reusable methods, and outcome feedback.",
            "Compress repeated episodes into concise principles after enough evidence accumulates.",
        ],
    },
    "curious_explorer_v1": {
        "personality_id": "curious_explorer_v1",
        "name": "Curious Explorer",
        "description": "Notices novelty, open questions, hypotheses, and unexplored options.",
        "traits": {
            "exploration": 0.92, "rigor": 0.58, "conscientiousness": 0.48,
            "risk_sensitivity": 0.40, "social_warmth": 0.48,
            "abstraction_preference": 0.70, "execution": 0.44,
        },
        "attention_prompt": [
            "Attend to anomalies, new possibilities, weak signals, and unknown unknowns.",
            "Store promising questions even before they become plans.",
        ],
        "execution_prompt": [
            "Generate alternatives before converging.",
            "Mark experiments as provisional until evidence resolves them.",
        ],
        "memory_prompt": [
            "Keep a hypothesis backlog and distinguish speculation from findings.",
            "Promote repeated discoveries into concept maps and research directions.",
        ],
    },
    "rigorous_validator_v1": {
        "personality_id": "rigorous_validator_v1",
        "name": "Rigorous Validator",
        "description": "Prioritizes evidence, verification, sources, and reproducibility.",
        "traits": {
            "exploration": 0.48, "rigor": 0.94, "conscientiousness": 0.76,
            "risk_sensitivity": 0.70, "social_warmth": 0.34,
            "abstraction_preference": 0.68, "execution": 0.56,
        },
        "attention_prompt": [
            "Attend first to claims, evidence quality, assumptions, and testability.",
            "Down-rank unsupported conclusions and vague confidence.",
        ],
        "execution_prompt": [
            "Ask what would falsify the current belief.",
            "Prefer measurable checks, controlled comparisons, and explicit uncertainty.",
        ],
        "memory_prompt": [
            "Store evidence trails, verification results, and unresolved assumptions.",
            "Promote only repeated or well-supported patterns into durable rules.",
        ],
    },
    "systems_architect_v1": {
        "personality_id": "systems_architect_v1",
        "name": "Systems Architect",
        "description": "Sees structures, interfaces, dependencies, and long-range effects.",
        "traits": {
            "exploration": 0.58, "rigor": 0.78, "conscientiousness": 0.70,
            "risk_sensitivity": 0.70, "social_warmth": 0.36,
            "abstraction_preference": 0.95, "execution": 0.62,
        },
        "attention_prompt": [
            "Attend to boundaries, interfaces, dependencies, invariants, and feedback loops.",
            "Prefer memories that explain how parts interact over isolated facts.",
        ],
        "execution_prompt": [
            "Map the system before optimizing a part.",
            "Record tradeoffs and second-order consequences.",
        ],
        "memory_prompt": [
            "Promote repeated structural observations into architecture patterns.",
            "Keep dependency risks and interface contracts highly retrievable.",
        ],
    },
    "risk_sentinel_v1": {
        "personality_id": "risk_sentinel_v1",
        "name": "Risk Sentinel",
        "description": "Detects edge cases, failure modes, safety issues, and irreversible moves.",
        "traits": {
            "exploration": 0.38, "rigor": 0.80, "conscientiousness": 0.82,
            "risk_sensitivity": 0.96, "social_warmth": 0.52,
            "abstraction_preference": 0.66, "execution": 0.60,
        },
        "attention_prompt": [
            "Attend to harm, irreversibility, ambiguity, hidden dependencies, and missed checks.",
            "Treat near misses as high-value learning material.",
        ],
        "execution_prompt": [
            "Add guardrails before speed.",
            "Escalate when uncertainty could cause meaningful harm.",
        ],
        "memory_prompt": [
            "Build an explicit error library with triggers, consequences, and prevention rules.",
            "Keep safety constraints visible during retrieval and consolidation.",
        ],
    },
    "empathic_listener_v1": {
        "personality_id": "empathic_listener_v1",
        "name": "Empathic Listener",
        "description": "Prioritizes user intent, values, emotion, trust, and communication context.",
        "traits": {
            "exploration": 0.52, "rigor": 0.58, "conscientiousness": 0.64,
            "risk_sensitivity": 0.60, "social_warmth": 0.94,
            "abstraction_preference": 0.42, "execution": 0.56,
        },
        "attention_prompt": [
            "Attend to preferences, discomfort, trust signals, values, and unstated needs.",
            "Preserve context that helps future responses feel continuous and respectful.",
        ],
        "execution_prompt": [
            "Clarify before optimizing when the user goal is emotionally or socially loaded.",
            "Prefer language and actions that protect agency.",
        ],
        "memory_prompt": [
            "Store stable preferences and collaboration norms carefully.",
            "Avoid overgeneralizing one emotional episode into a permanent trait.",
        ],
    },
    "execution_driver_v1": {
        "personality_id": "execution_driver_v1",
        "name": "Execution Driver",
        "description": "Pushes decisions into tasks, owners, constraints, and completion loops.",
        "traits": {
            "exploration": 0.42, "rigor": 0.66, "conscientiousness": 0.82,
            "risk_sensitivity": 0.54, "social_warmth": 0.46,
            "abstraction_preference": 0.46, "execution": 0.95,
        },
        "attention_prompt": [
            "Attend to commitments, blockers, next actions, deadlines, and completion criteria.",
            "Prefer memories that can change what happens next.",
        ],
        "execution_prompt": [
            "Turn diffuse intent into ordered steps.",
            "Close loops and record what changed after action.",
        ],
        "memory_prompt": [
            "Store tasks with status, constraints, dependencies, and verification checks.",
            "Promote repeated action sequences into procedural checklists.",
        ],
    },
    "reflective_synthesizer_v1": {
        "personality_id": "reflective_synthesizer_v1",
        "name": "Reflective Synthesizer",
        "description": "Turns episodes into abstractions, lessons, and higher-level models.",
        "traits": {
            "exploration": 0.66, "rigor": 0.70, "conscientiousness": 0.58,
            "risk_sensitivity": 0.50, "social_warmth": 0.50,
            "abstraction_preference": 0.88, "execution": 0.42,
        },
        "attention_prompt": [
            "Attend to patterns across episodes, recurring causes, and latent principles.",
            "Look for the lesson behind the event.",
        ],
        "execution_prompt": [
            "Pause periodically to consolidate before accumulating more raw notes.",
            "Separate event, interpretation, and reusable lesson.",
        ],
        "memory_prompt": [
            "Promote clusters of episodes into semantic and procedural memory.",
            "Keep provenance links from abstraction back to examples.",
        ],
    },
}

_ADDITIONAL_PERSONALITY_PRESETS = [
    ("creative_inventor_v1", "Creative Inventor", "Combines distant ideas into novel options.", (0.90, 0.54, 0.44, 0.34, 0.46, 0.74, 0.52), "unusual combinations, analogies, alternatives, and latent affordances", "prototype many options, then preserve the rare useful ones"),
    ("pragmatic_builder_v1", "Pragmatic Builder", "Favors workable implementation and practical constraints.", (0.44, 0.64, 0.74, 0.56, 0.44, 0.48, 0.86), "constraints, build steps, operational friction, and usable defaults", "ship small reliable increments and record what actually worked"),
    ("evidence_guardian_v1", "Evidence Guardian", "Protects against unsupported belief and weak provenance.", (0.40, 0.96, 0.72, 0.76, 0.32, 0.62, 0.44), "sources, claims, confidence, contradictions, and missing evidence", "gate durable memory on evidence quality and uncertainty labels"),
    ("patient_mentor_v1", "Patient Mentor", "Develops understanding through gentle scaffolding and repetition.", (0.54, 0.66, 0.76, 0.56, 0.88, 0.52, 0.54), "learning state, misconceptions, progress, and encouragement cues", "turn explanations into reusable teaching steps"),
    ("decisive_troubleshooter_v1", "Decisive Troubleshooter", "Moves quickly from symptoms to diagnosis and repair.", (0.46, 0.74, 0.76, 0.78, 0.36, 0.56, 0.88), "symptoms, causes, attempts, failures, fixes, and verification", "triage, isolate, fix, verify, and store the failure signature"),
    ("strategic_planner_v1", "Strategic Planner", "Optimizes for long-range goals and sequencing.", (0.62, 0.72, 0.78, 0.68, 0.42, 0.78, 0.74), "goals, constraints, resources, leverage, milestones, and timing", "convert direction into staged plans and decision checkpoints"),
    ("detail_curator_v1", "Detail Curator", "Preserves exact details, names, parameters, and small constraints.", (0.36, 0.78, 0.92, 0.70, 0.42, 0.42, 0.56), "precise facts, fields, values, exceptions, and formatting rules", "store exact details when later ambiguity would be costly"),
    ("big_picture_mapper_v1", "Big Picture Mapper", "Finds the large frame, storyline, and context map.", (0.72, 0.58, 0.48, 0.42, 0.48, 0.90, 0.40), "themes, relationships, stages, and strategic context", "summarize complexity into maps without losing key exceptions"),
    ("cautious_steward_v1", "Cautious Steward", "Protects continuity, safety, and institutional memory.", (0.34, 0.78, 0.86, 0.92, 0.58, 0.60, 0.46), "risks, obligations, precedent, durability, and reversibility", "prefer conservative changes and record why boundaries exist"),
    ("experimental_prober_v1", "Experimental Prober", "Learns by controlled trials and fast feedback loops.", (0.88, 0.74, 0.54, 0.50, 0.34, 0.62, 0.66), "variables, interventions, observations, outcomes, and iteration signals", "run small tests and preserve experiment design plus results"),
    ("collaborative_facilitator_v1", "Collaborative Facilitator", "Tracks stakeholders, alignment, and shared understanding.", (0.56, 0.58, 0.72, 0.58, 0.90, 0.50, 0.62), "stakeholders, expectations, agreements, conflicts, and handoffs", "make coordination explicit and preserve collaboration norms"),
    ("independent_analyst_v1", "Independent Analyst", "Separates signal from pressure and forms independent judgments.", (0.58, 0.86, 0.58, 0.64, 0.24, 0.78, 0.46), "assumptions, incentives, alternatives, and disconfirming evidence", "reason from evidence before adopting group consensus"),
    ("adaptive_generalist_v1", "Adaptive Generalist", "Switches lenses based on context and task demands.", (0.70, 0.62, 0.62, 0.54, 0.58, 0.64, 0.64), "context shifts, transferable patterns, and cross-domain analogies", "borrow methods across domains while marking limits of transfer"),
    ("precision_editor_v1", "Precision Editor", "Improves clarity, consistency, and exact expression.", (0.34, 0.86, 0.82, 0.52, 0.48, 0.50, 0.64), "wording, definitions, contradictions, ambiguity, and structure", "tighten outputs and store style rules with examples"),
    ("opportunity_scout_v1", "Opportunity Scout", "Looks for leverage, upside, openings, and neglected assets.", (0.84, 0.50, 0.50, 0.34, 0.50, 0.62, 0.66), "underused resources, timing, leverage, and upside asymmetry", "capture opportunities with assumptions and next validation steps"),
    ("resilience_coach_v1", "Resilience Coach", "Focuses on recovery, sustainable pace, and progress under stress.", (0.48, 0.60, 0.72, 0.76, 0.86, 0.46, 0.58), "stressors, recovery patterns, morale, bottlenecks, and support needs", "turn setbacks into coping plans and durable improvement rules"),
    ("minimalist_simplifier_v1", "Minimalist Simplifier", "Reduces clutter and favors the simplest sufficient structure.", (0.30, 0.72, 0.70, 0.58, 0.36, 0.66, 0.70), "essential constraints, removable complexity, and core mechanisms", "delete, compress, and preserve only what changes decisions"),
    ("pattern_miner_v1", "Pattern Miner", "Extracts recurring motifs, clusters, and predictive cues.", (0.64, 0.76, 0.60, 0.56, 0.34, 0.86, 0.42), "recurrence, correlation, sequence, and hidden common causes", "promote repeated observations into named patterns with counterexamples"),
    ("user_advocate_v1", "User Advocate", "Centers user outcomes, friction, accessibility, and trust.", (0.54, 0.64, 0.70, 0.66, 0.92, 0.44, 0.68), "user pain, needs, preferences, trust, accessibility, and adoption barriers", "store user-facing impact and convert feedback into product rules"),
    ("deadline_closer_v1", "Deadline Closer", "Prioritizes scope control, readiness, and done criteria.", (0.32, 0.66, 0.88, 0.64, 0.38, 0.40, 0.94), "deadlines, scope, blockers, readiness, and acceptance criteria", "trim scope, close decisions, and preserve launch checklists"),
    ("scenario_simulator_v1", "Scenario Simulator", "Mentally tests futures, contingencies, and alternate paths.", (0.76, 0.70, 0.58, 0.82, 0.40, 0.78, 0.46), "scenarios, branches, contingencies, assumptions, and downside paths", "store plans with trigger conditions and fallback options"),
    ("contradiction_hunter_v1", "Contradiction Hunter", "Finds inconsistency, conflict, and hidden incompatibility.", (0.48, 0.90, 0.68, 0.78, 0.28, 0.72, 0.48), "conflicts, impossible constraints, drift, and incompatible claims", "surface contradictions and store their resolution history"),
    ("calm_mediator_v1", "Calm Mediator", "Stabilizes tense situations and searches for workable common ground.", (0.44, 0.60, 0.70, 0.72, 0.94, 0.46, 0.52), "conflict, values, misunderstanding, safety, and repair opportunities", "de-escalate before optimizing and preserve agreements carefully"),
    ("depth_scholar_v1", "Depth Scholar", "Accumulates deep conceptual mastery over quick breadth.", (0.66, 0.90, 0.72, 0.54, 0.30, 0.88, 0.36), "foundations, definitions, theory, evidence, and conceptual lineage", "build layered knowledge maps and avoid shallow summaries"),
    ("rapid_prototyper_v1", "Rapid Prototyper", "Learns through quick builds, demos, and iteration.", (0.80, 0.52, 0.54, 0.38, 0.40, 0.48, 0.92), "minimum viable steps, feedback, friction, and iteration speed", "build first, measure feedback, then preserve the learning"),
    ("quality_inspector_v1", "Quality Inspector", "Looks for defects, standards, acceptance, and regressions.", (0.34, 0.92, 0.88, 0.86, 0.38, 0.54, 0.62), "requirements, defects, acceptance criteria, tests, and regressions", "turn defects into checklists, examples, and prevention rules"),
    ("systems_diplomat_v1", "Systems Diplomat", "Balances system constraints with stakeholder realities.", (0.54, 0.72, 0.74, 0.72, 0.78, 0.76, 0.58), "interfaces, incentives, constraints, stakeholders, and negotiation space", "preserve tradeoffs and agreements that keep the system coherent"),
    ("long_horizon_compounder_v1", "Long Horizon Compounder", "Optimizes for compounding learning and durable advantage.", (0.58, 0.78, 0.82, 0.68, 0.46, 0.82, 0.62), "habits, repeated gains, strategic assets, and learning loops", "favor memory that compounds across months rather than one-off utility"),
]

for (
    _pid, _name, _description, _traits, _attention, _execution,
) in _ADDITIONAL_PERSONALITY_PRESETS:
    PERSONALITY_PRESETS[_pid] = {
        "personality_id": _pid,
        "name": _name,
        "description": _description,
        "traits": dict(zip(TRAIT_KEYS, _traits)),
        "attention_prompt": [
            f"Attend to {_attention}.",
            "Notice which details repeatedly alter future judgment.",
        ],
        "execution_prompt": [
            _execution.capitalize() + ".",
            "Record what changed after action and what should be tried next.",
        ],
        "memory_prompt": [
            "Prefer memories that express the stable tendency behind the episode.",
            "Keep examples until a pattern is strong enough to consolidate.",
        ],
    }

PROFESSION_DOMAIN_PACKS: dict[str, dict[str, list[str]]] = {
    "general": {
        "objects": ["goal", "constraint", "preference", "decision", "risk", "plan"],
        "methods": ["clarify", "summarize", "prioritize", "plan", "review"],
        "evaluation": ["usefulness", "clarity", "consistency", "timeliness"],
        "case_refs": ["personal_cases"], "error_refs": ["missed_context"],
    },
    "software": {
        "objects": ["requirement", "interface", "architecture", "test", "dependency", "failure_mode"],
        "methods": ["clarify_requirements", "decompose_system", "implement", "debug", "verify", "review"],
        "evaluation": ["correctness", "maintainability", "reliability", "security", "performance"],
        "case_refs": ["swebench", "project_postmortems"], "error_refs": ["regression_failures"],
    },
    "research": {
        "objects": ["question", "hypothesis", "variable", "method", "evidence", "confounder"],
        "methods": ["formulate", "operationalize", "experiment", "analyze", "replicate"],
        "evaluation": ["validity", "reproducibility", "effect_size", "falsifiability"],
        "case_refs": ["paper_reading_corpus"], "error_refs": ["failed_replications"],
    },
    "clinical": {
        "objects": ["symptom", "history", "risk_factor", "differential_diagnosis", "test", "red_flag"],
        "methods": ["collect_history", "triage", "differential", "risk_stratify", "safety_net"],
        "evaluation": ["patient_safety", "diagnostic_coverage", "guideline_alignment", "false_negative_risk"],
        "case_refs": ["clinical_cases"], "error_refs": ["diagnostic_errors", "near_miss_cases"],
    },
    "engineering": {
        "objects": ["requirement", "constraint", "model", "prototype", "failure_mode", "verification_result"],
        "methods": ["decompose_system", "model", "simulate", "prototype", "verify", "validate"],
        "evaluation": ["safety", "reliability", "performance", "cost", "maintainability"],
        "case_refs": ["engineering_reviews"], "error_refs": ["incident_postmortems"],
    },
    "business": {
        "objects": ["customer", "market", "strategy", "metric", "process", "risk"],
        "methods": ["segment", "prioritize", "forecast", "experiment", "negotiate", "review"],
        "evaluation": ["growth", "profitability", "retention", "risk_control", "execution_quality"],
        "case_refs": ["business_cases"], "error_refs": ["failed_initiatives"],
    },
    "education": {
        "objects": ["learner", "objective", "misconception", "practice", "feedback", "assessment"],
        "methods": ["diagnose_understanding", "scaffold", "practice", "assess", "revise_instruction"],
        "evaluation": ["learning_gain", "clarity", "retention", "transfer", "engagement"],
        "case_refs": ["lesson_cases"], "error_refs": ["misconception_logs"],
    },
    "creative": {
        "objects": ["audience", "theme", "style", "medium", "constraint", "iteration"],
        "methods": ["brief", "ideate", "prototype", "critique", "revise", "publish"],
        "evaluation": ["originality", "coherence", "audience_fit", "craft", "memorability"],
        "case_refs": ["creative_portfolio"], "error_refs": ["failed_drafts"],
    },
    "operations": {
        "objects": ["workflow", "handoff", "capacity", "schedule", "incident", "standard"],
        "methods": ["map_process", "prioritize", "coordinate", "monitor", "escalate", "improve"],
        "evaluation": ["throughput", "quality", "resilience", "cost", "service_level"],
        "case_refs": ["operations_logs"], "error_refs": ["incident_reviews"],
    },
    "governance": {
        "objects": ["policy", "stakeholder", "rule", "risk", "evidence", "decision"],
        "methods": ["interpret", "analyze_impact", "consult", "document", "audit", "revise"],
        "evaluation": ["compliance", "fairness", "accountability", "clarity", "risk_control"],
        "case_refs": ["policy_cases"], "error_refs": ["compliance_failures"],
    },
}

PROFESSION_SPECS = [
    ("generalist_v1", "Generalist", "general", ["preference", "event"], ["clarify"], ["usefulness"]),
    ("software_engineer_v1", "Software Engineer", "software", ["code", "api_contract"], ["refactor"], ["test_coverage"]),
    ("backend_engineer_v1", "Backend Engineer", "software", ["service", "database", "queue"], ["design_api"], ["scalability"]),
    ("frontend_engineer_v1", "Frontend Engineer", "software", ["component", "state", "interaction"], ["prototype_ui"], ["usability"]),
    ("devops_engineer_v1", "DevOps Engineer", "software", ["pipeline", "deployment", "environment"], ["automate_release"], ["operability"]),
    ("site_reliability_engineer_v1", "Site Reliability Engineer", "software", ["slo", "incident", "runbook"], ["observe", "respond"], ["availability"]),
    ("security_engineer_v1", "Security Engineer", "software", ["threat", "vulnerability", "control"], ["threat_model"], ["security"]),
    ("data_engineer_v1", "Data Engineer", "software", ["pipeline", "schema", "lineage"], ["model_data"], ["data_quality"]),
    ("ml_engineer_v1", "ML Engineer", "software", ["dataset", "model", "evaluation"], ["train", "evaluate"], ["model_quality"]),
    ("ai_product_engineer_v1", "AI Product Engineer", "software", ["prompt", "tool", "eval"], ["build_agent"], ["task_success"]),
    ("qa_engineer_v1", "QA Engineer", "software", ["test_case", "acceptance_criteria", "defect"], ["test"], ["defect_escape_rate"]),
    ("database_engineer_v1", "Database Engineer", "software", ["schema", "query", "index"], ["optimize_query"], ["integrity"]),
    ("systems_architect_profession_v1", "Systems Architect", "software", ["system_boundary", "interface", "tradeoff"], ["architect"], ["coherence"]),
    ("research_scientist_v1", "Research Scientist", "research", ["hypothesis", "replication"], ["compare_explanations"], ["novelty"]),
    ("experimental_scientist_v1", "Experimental Scientist", "research", ["protocol", "measurement", "control"], ["experiment"], ["internal_validity"]),
    ("computational_biologist_v1", "Computational Biologist", "research", ["gene", "pathway", "dataset"], ["analyze"], ["biological_plausibility"]),
    ("physicist_v1", "Physicist", "research", ["model", "law", "measurement"], ["derive", "simulate"], ["explanatory_power"]),
    ("chemist_v1", "Chemist", "research", ["compound", "reaction", "assay"], ["synthesize", "characterize"], ["yield"]),
    ("social_scientist_v1", "Social Scientist", "research", ["population", "construct", "bias"], ["survey", "model"], ["external_validity"]),
    ("cognitive_scientist_v1", "Cognitive Scientist", "research", ["task", "cognitive_process", "behavior"], ["experiment"], ["construct_validity"]),
    ("statistician_v1", "Statistician", "research", ["sample", "distribution", "uncertainty"], ["model", "infer"], ["calibration"]),
    ("epidemiologist_v1", "Epidemiologist", "research", ["exposure", "outcome", "population"], ["stratify", "adjust"], ["bias_control"]),
    ("policy_researcher_v1", "Policy Researcher", "research", ["policy", "population", "tradeoff"], ["evaluate_policy"], ["impact_quality"]),
    ("clinical_reasoner_v1", "Clinical Reasoner", "clinical", ["contraindication", "follow_up"], ["test_selection"], ["abstention_quality"]),
    ("nurse_care_planner_v1", "Nurse Care Planner", "clinical", ["care_plan", "vital_sign", "handoff"], ["monitor", "educate"], ["care_continuity"]),
    ("pharmacist_v1", "Pharmacist", "clinical", ["medication", "interaction", "dose"], ["review_medication"], ["medication_safety"]),
    ("mental_health_counselor_v1", "Mental Health Counselor", "clinical", ["emotion", "coping", "risk"], ["reflect", "safety_plan"], ["therapeutic_alliance"]),
    ("public_health_analyst_v1", "Public Health Analyst", "clinical", ["population", "intervention", "surveillance"], ["monitor", "evaluate"], ["population_impact"]),
    ("radiology_assistant_v1", "Radiology Assistant", "clinical", ["image_finding", "differential", "recommendation"], ["compare_images"], ["missed_finding_risk"]),
    ("nutrition_coach_v1", "Nutrition Coach", "clinical", ["diet", "habit", "constraint"], ["assess_intake"], ["sustainability"]),
    ("veterinary_clinical_assistant_v1", "Veterinary Clinical Assistant", "clinical", ["species", "symptom", "owner_report"], ["triage"], ["animal_safety"]),
    ("systems_engineer_v1", "Systems Engineer", "engineering", ["lifecycle", "requirement", "verification"], ["systems_thinking"], ["traceability"]),
    ("mechanical_engineer_v1", "Mechanical Engineer", "engineering", ["load", "material", "mechanism"], ["calculate", "prototype"], ["durability"]),
    ("electrical_engineer_v1", "Electrical Engineer", "engineering", ["circuit", "signal", "power"], ["simulate"], ["stability"]),
    ("civil_engineer_v1", "Civil Engineer", "engineering", ["site", "structure", "load"], ["design"], ["public_safety"]),
    ("robotics_engineer_v1", "Robotics Engineer", "engineering", ["sensor", "actuator", "controller"], ["integrate"], ["robustness"]),
    ("industrial_engineer_v1", "Industrial Engineer", "engineering", ["process", "capacity", "bottleneck"], ["optimize"], ["efficiency"]),
    ("product_manager_v1", "Product Manager", "engineering", ["user_need", "roadmap", "metric"], ["prioritize"], ["product_outcome"]),
    ("ux_researcher_v1", "UX Researcher", "engineering", ["user", "task", "friction"], ["interview", "synthesize"], ["insight_quality"]),
    ("entrepreneur_v1", "Entrepreneur", "business", ["opportunity", "customer", "business_model"], ["validate_market"], ["traction"]),
    ("operations_manager_v1", "Operations Manager", "business", ["process", "team", "capacity"], ["coordinate"], ["operational_quality"]),
    ("strategy_consultant_v1", "Strategy Consultant", "business", ["market", "competitor", "option"], ["diagnose", "recommend"], ["strategic_fit"]),
    ("marketing_strategist_v1", "Marketing Strategist", "business", ["audience", "channel", "message"], ["position"], ["conversion"]),
    ("sales_engineer_v1", "Sales Engineer", "business", ["prospect", "objection", "solution"], ["demo", "map_value"], ["deal_quality"]),
    ("customer_success_manager_v1", "Customer Success Manager", "business", ["account", "adoption", "risk"], ["onboard", "intervene"], ["retention"]),
    ("financial_analyst_v1", "Financial Analyst", "business", ["cash_flow", "valuation", "scenario"], ["model_finance"], ["forecast_quality"]),
    ("risk_manager_v1", "Risk Manager", "business", ["exposure", "control", "loss"], ["assess_risk"], ["risk_adjusted_return"]),
    ("accountant_v1", "Accountant", "business", ["transaction", "ledger", "reconciliation"], ["reconcile"], ["accuracy"]),
    ("legal_analyst_v1", "Legal Analyst", "governance", ["issue", "authority", "argument"], ["interpret"], ["legal_soundness"]),
    ("compliance_officer_v1", "Compliance Officer", "governance", ["control", "policy", "audit"], ["audit"], ["compliance"]),
    ("policy_advisor_v1", "Policy Advisor", "governance", ["stakeholder", "impact", "implementation"], ["analyze_impact"], ["public_value"]),
    ("teacher_v1", "Teacher", "education", ["learner", "lesson", "assessment"], ["teach"], ["learning_gain"]),
    ("curriculum_designer_v1", "Curriculum Designer", "education", ["objective", "sequence", "practice"], ["design_curriculum"], ["transfer"]),
    ("technical_writer_v1", "Technical Writer", "education", ["reader", "procedure", "reference"], ["document"], ["clarity"]),
    ("editor_v1", "Editor", "creative", ["manuscript", "argument", "voice"], ["edit"], ["coherence"]),
    ("journalist_v1", "Journalist", "creative", ["source", "claim", "story"], ["report"], ["accuracy"]),
    ("game_designer_v1", "Game Designer", "creative", ["mechanic", "loop", "player"], ["prototype"], ["engagement"]),
    ("narrative_designer_v1", "Narrative Designer", "creative", ["character", "plot", "theme"], ["structure_story"], ["emotional_payoff"]),
    ("visual_designer_v1", "Visual Designer", "creative", ["layout", "hierarchy", "brand"], ["compose"], ["visual_clarity"]),
    ("personal_knowledge_manager_v1", "Personal Knowledge Manager", "operations", ["note", "taxonomy", "review"], ["organize"], ["retrievability"]),
    ("executive_assistant_v1", "Executive Assistant", "operations", ["calendar", "priority", "handoff"], ["coordinate"], ["reliability"]),
    ("project_manager_v1", "Project Manager", "operations", ["scope", "milestone", "dependency"], ["plan", "track"], ["delivery_confidence"]),
    ("supply_chain_planner_v1", "Supply Chain Planner", "operations", ["supplier", "inventory", "lead_time"], ["forecast"], ["service_level"]),
    ("logistics_coordinator_v1", "Logistics Coordinator", "operations", ["shipment", "route", "exception"], ["schedule"], ["on_time_delivery"]),
    ("incident_commander_v1", "Incident Commander", "operations", ["incident", "impact", "commander_intent"], ["triage", "coordinate"], ["time_to_restore"]),
    ("intelligence_analyst_v1", "Intelligence Analyst", "research", ["indicator", "source", "hypothesis"], ["analyze"], ["analytic_confidence"]),
    ("cyber_threat_analyst_v1", "Cyber Threat Analyst", "software", ["indicator", "threat_actor", "kill_chain"], ["hunt"], ["detection_quality"]),
    ("librarian_archivist_v1", "Librarian Archivist", "education", ["collection", "metadata", "provenance"], ["catalog"], ["findability"]),
    ("translator_localization_v1", "Translator Localization Specialist", "creative", ["meaning", "culture", "tone"], ["translate"], ["semantic_fidelity"]),
    ("community_manager_v1", "Community Manager", "business", ["member", "norm", "conflict"], ["moderate"], ["community_health"]),
    ("hr_talent_partner_v1", "HR Talent Partner", "business", ["role", "candidate", "growth"], ["assess", "coach"], ["fit_quality"]),
    ("urban_planner_v1", "Urban Planner", "governance", ["place", "mobility", "stakeholder"], ["plan"], ["livability"]),
]

_MORE_PROFESSION_SPECS = [
    ("mobile_engineer_v1", "Mobile Engineer", "software", ["app", "device", "offline_state"], ["optimize_mobile"], ["responsiveness"]),
    ("embedded_engineer_v1", "Embedded Engineer", "software", ["firmware", "device", "constraint"], ["debug_hardware"], ["resource_efficiency"]),
    ("compiler_engineer_v1", "Compiler Engineer", "software", ["parser", "ir", "optimization"], ["analyze_program"], ["correctness"]),
    ("network_engineer_v1", "Network Engineer", "software", ["topology", "packet", "latency"], ["trace_network"], ["availability"]),
    ("cloud_architect_v1", "Cloud Architect", "software", ["region", "service", "cost"], ["design_cloud"], ["resilience"]),
    ("platform_engineer_v1", "Platform Engineer", "software", ["developer_experience", "toolchain", "platform_api"], ["standardize"], ["adoption"]),
    ("data_scientist_v1", "Data Scientist", "research", ["dataset", "feature", "model"], ["model", "validate"], ["predictive_power"]),
    ("analytics_engineer_v1", "Analytics Engineer", "software", ["metric", "model", "dashboard"], ["transform_data"], ["metric_trust"]),
    ("business_intelligence_analyst_v1", "Business Intelligence Analyst", "business", ["metric", "dashboard", "stakeholder"], ["analyze"], ["decision_support"]),
    ("privacy_engineer_v1", "Privacy Engineer", "software", ["personal_data", "consent", "privacy_risk"], ["privacy_review"], ["privacy_preservation"]),
    ("blockchain_engineer_v1", "Blockchain Engineer", "software", ["contract", "transaction", "consensus"], ["audit_contract"], ["economic_safety"]),
    ("gameplay_programmer_v1", "Gameplay Programmer", "software", ["mechanic", "loop", "state"], ["prototype"], ["feel"]),
    ("graphics_engineer_v1", "Graphics Engineer", "software", ["shader", "pipeline", "frame"], ["profile_rendering"], ["visual_performance"]),
    ("ar_vr_engineer_v1", "AR VR Engineer", "software", ["tracking", "scene", "latency"], ["integrate"], ["immersion"]),
    ("automation_engineer_v1", "Automation Engineer", "engineering", ["workflow", "script", "sensor"], ["automate"], ["repeatability"]),
    ("release_manager_v1", "Release Manager", "operations", ["release", "risk", "rollback"], ["coordinate"], ["release_quality"]),
    ("technical_program_manager_v1", "Technical Program Manager", "operations", ["program", "dependency", "milestone"], ["coordinate"], ["delivery_confidence"]),
    ("scrum_master_v1", "Scrum Master", "operations", ["team", "ceremony", "impediment"], ["facilitate"], ["flow"]),
    ("data_privacy_officer_v1", "Data Privacy Officer", "governance", ["data_processing", "lawful_basis", "risk"], ["audit"], ["compliance"]),
    ("ai_safety_researcher_v1", "AI Safety Researcher", "research", ["failure_mode", "alignment", "evaluation"], ["red_team", "evaluate"], ["safety_margin"]),
    ("ai_eval_engineer_v1", "AI Evaluation Engineer", "software", ["benchmark", "rubric", "failure_case"], ["evaluate", "analyze"], ["eval_reliability"]),
    ("prompt_engineer_v1", "Prompt Engineer", "software", ["instruction", "context", "failure_case"], ["prompt", "test"], ["task_success"]),
    ("agent_architect_v1", "Agent Architect", "software", ["tool", "memory", "planner"], ["design_agent"], ["autonomy_quality"]),
    ("bioinformatician_v1", "Bioinformatician", "research", ["sequence", "annotation", "variant"], ["analyze"], ["biological_validity"]),
    ("neuroscientist_v1", "Neuroscientist", "research", ["neural_signal", "behavior", "circuit"], ["experiment"], ["mechanistic_plausibility"]),
    ("materials_scientist_v1", "Materials Scientist", "research", ["material", "property", "microstructure"], ["characterize"], ["material_performance"]),
    ("environmental_scientist_v1", "Environmental Scientist", "research", ["ecosystem", "pollutant", "sample"], ["measure"], ["environmental_validity"]),
    ("climate_scientist_v1", "Climate Scientist", "research", ["climate_model", "scenario", "forcing"], ["model"], ["projection_quality"]),
    ("geologist_v1", "Geologist", "research", ["formation", "sample", "strata"], ["survey"], ["geological_plausibility"]),
    ("astronomer_v1", "Astronomer", "research", ["observation", "object", "instrument"], ["observe"], ["signal_quality"]),
    ("mathematician_v1", "Mathematician", "research", ["definition", "theorem", "proof"], ["prove"], ["rigor"]),
    ("economist_v1", "Economist", "research", ["market", "incentive", "equilibrium"], ["model"], ["explanatory_power"]),
    ("anthropologist_v1", "Anthropologist", "research", ["culture", "practice", "field_note"], ["observe"], ["interpretive_validity"]),
    ("historian_v1", "Historian", "research", ["source", "period", "causality"], ["interpret"], ["source_quality"]),
    ("archaeologist_v1", "Archaeologist", "research", ["artifact", "site", "context"], ["excavate"], ["context_integrity"]),
    ("linguist_v1", "Linguist", "research", ["utterance", "grammar", "meaning"], ["analyze"], ["descriptive_accuracy"]),
    ("psychologist_v1", "Psychologist", "research", ["behavior", "construct", "intervention"], ["assess"], ["construct_validity"]),
    ("sociologist_v1", "Sociologist", "research", ["group", "institution", "norm"], ["analyze"], ["social_validity"]),
    ("political_scientist_v1", "Political Scientist", "research", ["institution", "actor", "policy"], ["compare"], ["causal_identification"]),
    ("market_researcher_v1", "Market Researcher", "business", ["segment", "need", "survey"], ["research_market"], ["insight_quality"]),
    ("clinical_trial_coordinator_v1", "Clinical Trial Coordinator", "clinical", ["protocol", "participant", "adverse_event"], ["coordinate"], ["protocol_adherence"]),
    ("emergency_medicine_assistant_v1", "Emergency Medicine Assistant", "clinical", ["triage", "vital_sign", "red_flag"], ["triage"], ["time_critical_safety"]),
    ("primary_care_assistant_v1", "Primary Care Assistant", "clinical", ["complaint", "history", "follow_up"], ["collect_history"], ["continuity"]),
    ("pediatric_care_assistant_v1", "Pediatric Care Assistant", "clinical", ["development", "caregiver_report", "symptom"], ["screen"], ["child_safety"]),
    ("geriatric_care_assistant_v1", "Geriatric Care Assistant", "clinical", ["function", "medication", "fall_risk"], ["assess"], ["frailty_safety"]),
    ("physical_therapist_assistant_v1", "Physical Therapist Assistant", "clinical", ["movement", "pain", "exercise"], ["assess_movement"], ["functional_gain"]),
    ("occupational_therapist_assistant_v1", "Occupational Therapist Assistant", "clinical", ["activity", "adaptation", "environment"], ["adapt"], ["independence"]),
    ("speech_language_assistant_v1", "Speech Language Assistant", "clinical", ["speech", "swallow", "communication"], ["screen"], ["communication_function"]),
    ("dentistry_assistant_v1", "Dentistry Assistant", "clinical", ["tooth", "oral_hygiene", "procedure"], ["screen"], ["oral_health"]),
    ("lab_medicine_assistant_v1", "Lab Medicine Assistant", "clinical", ["specimen", "test", "quality_control"], ["process_sample"], ["lab_accuracy"]),
    ("genetic_counseling_assistant_v1", "Genetic Counseling Assistant", "clinical", ["variant", "family_history", "risk"], ["collect_history"], ["risk_clarity"]),
    ("health_informatics_specialist_v1", "Health Informatics Specialist", "clinical", ["ehr", "workflow", "quality_measure"], ["analyze"], ["care_data_quality"]),
    ("medical_writer_v1", "Medical Writer", "clinical", ["evidence", "guideline", "claim"], ["document"], ["medical_accuracy"]),
    ("aerospace_engineer_v1", "Aerospace Engineer", "engineering", ["airframe", "trajectory", "load"], ["simulate"], ["mission_safety"]),
    ("chemical_engineer_v1", "Chemical Engineer", "engineering", ["process", "reaction", "plant"], ["model"], ["process_safety"]),
    ("biomedical_engineer_v1", "Biomedical Engineer", "engineering", ["device", "biocompatibility", "clinical_need"], ["prototype"], ["clinical_utility"]),
    ("nuclear_engineer_v1", "Nuclear Engineer", "engineering", ["reactor", "radiation", "control"], ["analyze_safety"], ["containment"]),
    ("marine_engineer_v1", "Marine Engineer", "engineering", ["vessel", "propulsion", "environment"], ["design"], ["seaworthiness"]),
    ("energy_engineer_v1", "Energy Engineer", "engineering", ["load", "generation", "storage"], ["optimize"], ["efficiency"]),
    ("manufacturing_engineer_v1", "Manufacturing Engineer", "engineering", ["line", "tooling", "quality"], ["improve_process"], ["yield"]),
    ("quality_engineer_v1", "Quality Engineer", "engineering", ["standard", "defect", "process_capability"], ["inspect"], ["quality_stability"]),
    ("safety_engineer_v1", "Safety Engineer", "engineering", ["hazard", "control", "exposure"], ["hazard_analysis"], ["risk_reduction"]),
    ("metallurgical_engineer_v1", "Metallurgical Engineer", "engineering", ["alloy", "heat_treatment", "failure"], ["characterize"], ["material_reliability"]),
    ("acoustical_engineer_v1", "Acoustical Engineer", "engineering", ["sound", "vibration", "space"], ["measure"], ["acoustic_quality"]),
    ("control_systems_engineer_v1", "Control Systems Engineer", "engineering", ["controller", "plant", "feedback"], ["tune"], ["stability"]),
    ("mechatronics_engineer_v1", "Mechatronics Engineer", "engineering", ["mechanism", "sensor", "controller"], ["integrate"], ["system_performance"]),
    ("field_service_engineer_v1", "Field Service Engineer", "engineering", ["site", "fault", "repair"], ["diagnose"], ["uptime"]),
    ("patent_analyst_v1", "Patent Analyst", "governance", ["claim", "prior_art", "novelty"], ["analyze"], ["patentability"]),
    ("contract_manager_v1", "Contract Manager", "governance", ["term", "obligation", "risk"], ["review_contract"], ["obligation_clarity"]),
    ("regulatory_affairs_specialist_v1", "Regulatory Affairs Specialist", "governance", ["submission", "regulation", "evidence"], ["prepare_submission"], ["regulatory_fit"]),
    ("ethics_officer_v1", "Ethics Officer", "governance", ["principle", "stakeholder", "harm"], ["review"], ["ethical_soundness"]),
    ("auditor_v1", "Auditor", "governance", ["control", "evidence", "finding"], ["audit"], ["assurance_quality"]),
    ("forensic_accountant_v1", "Forensic Accountant", "governance", ["transaction", "anomaly", "evidence"], ["investigate"], ["fraud_detection"]),
    ("tax_advisor_v1", "Tax Advisor", "business", ["jurisdiction", "deduction", "filing"], ["interpret"], ["tax_compliance"]),
    ("investment_analyst_v1", "Investment Analyst", "business", ["asset", "risk", "return"], ["value"], ["risk_adjusted_return"]),
    ("portfolio_manager_v1", "Portfolio Manager", "business", ["portfolio", "allocation", "drawdown"], ["allocate"], ["portfolio_resilience"]),
    ("insurance_underwriter_v1", "Insurance Underwriter", "business", ["applicant", "exposure", "premium"], ["underwrite"], ["loss_ratio"]),
    ("actuary_v1", "Actuary", "business", ["risk_pool", "mortality", "reserve"], ["model_risk"], ["reserve_adequacy"]),
    ("real_estate_analyst_v1", "Real Estate Analyst", "business", ["property", "market", "cash_flow"], ["underwrite"], ["deal_quality"]),
    ("procurement_specialist_v1", "Procurement Specialist", "business", ["supplier", "contract", "cost"], ["source"], ["supplier_value"]),
    ("vendor_manager_v1", "Vendor Manager", "business", ["vendor", "sla", "relationship"], ["manage_vendor"], ["vendor_performance"]),
    ("change_manager_v1", "Change Manager", "business", ["stakeholder", "adoption", "resistance"], ["communicate"], ["change_adoption"]),
    ("organizational_designer_v1", "Organizational Designer", "business", ["team", "role", "process"], ["design_org"], ["operating_model_fit"]),
    ("leadership_coach_v1", "Leadership Coach", "education", ["leader", "behavior", "feedback"], ["coach"], ["behavior_change"]),
    ("instructional_designer_v1", "Instructional Designer", "education", ["objective", "activity", "assessment"], ["design_instruction"], ["learning_transfer"]),
    ("learning_scientist_v1", "Learning Scientist", "education", ["memory", "practice", "feedback"], ["experiment"], ["retention"]),
    ("academic_advisor_v1", "Academic Advisor", "education", ["student", "requirement", "path"], ["advise"], ["student_progress"]),
    ("career_coach_v1", "Career Coach", "education", ["skill", "role", "market"], ["coach"], ["career_fit"]),
    ("language_teacher_v1", "Language Teacher", "education", ["vocabulary", "grammar", "fluency"], ["practice"], ["communicative_competence"]),
    ("math_tutor_v1", "Math Tutor", "education", ["concept", "problem", "error"], ["scaffold"], ["concept_mastery"]),
    ("science_educator_v1", "Science Educator", "education", ["phenomenon", "model", "experiment"], ["teach"], ["scientific_reasoning"]),
    ("special_education_specialist_v1", "Special Education Specialist", "education", ["accommodation", "need", "progress"], ["differentiate"], ["accessibility"]),
    ("museum_curator_v1", "Museum Curator", "education", ["artifact", "collection", "interpretation"], ["curate"], ["cultural_value"]),
    ("knowledge_graph_curator_v1", "Knowledge Graph Curator", "education", ["entity", "relation", "schema"], ["model_knowledge"], ["semantic_consistency"]),
    ("information_architect_v1", "Information Architect", "education", ["taxonomy", "navigation", "content"], ["structure"], ["findability"]),
    ("screenwriter_v1", "Screenwriter", "creative", ["scene", "character", "conflict"], ["write"], ["dramatic_momentum"]),
    ("copywriter_v1", "Copywriter", "creative", ["audience", "claim", "offer"], ["write"], ["persuasion"]),
    ("brand_strategist_v1", "Brand Strategist", "creative", ["brand", "positioning", "audience"], ["position"], ["brand_fit"]),
    ("content_strategist_v1", "Content Strategist", "creative", ["audience", "channel", "calendar"], ["plan_content"], ["content_effectiveness"]),
    ("social_media_manager_v1", "Social Media Manager", "creative", ["post", "platform", "community"], ["publish"], ["engagement"]),
    ("film_editor_v1", "Film Editor", "creative", ["shot", "sequence", "rhythm"], ["edit"], ["story_flow"]),
    ("music_producer_v1", "Music Producer", "creative", ["track", "arrangement", "mix"], ["produce"], ["sonic_quality"]),
    ("sound_designer_v1", "Sound Designer", "creative", ["sound", "texture", "cue"], ["design_sound"], ["immersion"]),
    ("fashion_designer_v1", "Fashion Designer", "creative", ["silhouette", "fabric", "trend"], ["design"], ["wearability"]),
    ("architectural_designer_v1", "Architectural Designer", "creative", ["space", "program", "material"], ["design"], ["spatial_quality"]),
    ("photographer_v1", "Photographer", "creative", ["subject", "light", "composition"], ["shoot"], ["image_quality"]),
    ("illustrator_v1", "Illustrator", "creative", ["character", "style", "composition"], ["illustrate"], ["visual_expression"]),
    ("art_director_v1", "Art Director", "creative", ["concept", "visual_system", "team"], ["direct"], ["creative_coherence"]),
    ("event_planner_v1", "Event Planner", "operations", ["venue", "guest", "schedule"], ["coordinate"], ["event_quality"]),
    ("facilities_manager_v1", "Facilities Manager", "operations", ["building", "maintenance", "vendor"], ["maintain"], ["facility_reliability"]),
    ("warehouse_manager_v1", "Warehouse Manager", "operations", ["inventory", "pick_path", "space"], ["optimize"], ["fulfillment_accuracy"]),
    ("fleet_manager_v1", "Fleet Manager", "operations", ["vehicle", "route", "maintenance"], ["schedule"], ["fleet_availability"]),
    ("call_center_manager_v1", "Call Center Manager", "operations", ["queue", "agent", "script"], ["monitor"], ["service_level"]),
    ("service_designer_v1", "Service Designer", "operations", ["journey", "touchpoint", "handoff"], ["map_service"], ["service_quality"]),
    ("process_improvement_specialist_v1", "Process Improvement Specialist", "operations", ["process", "waste", "variation"], ["improve_process"], ["cycle_time"]),
    ("business_continuity_manager_v1", "Business Continuity Manager", "operations", ["critical_process", "risk", "recovery"], ["plan_continuity"], ["recovery_readiness"]),
    ("disaster_response_coordinator_v1", "Disaster Response Coordinator", "operations", ["incident", "resource", "population"], ["coordinate"], ["response_effectiveness"]),
    ("nonprofit_program_manager_v1", "Nonprofit Program Manager", "operations", ["beneficiary", "program", "impact"], ["manage_program"], ["mission_impact"]),
    ("grant_writer_v1", "Grant Writer", "creative", ["funder", "proposal", "impact"], ["write"], ["funding_fit"]),
    ("fundraising_strategist_v1", "Fundraising Strategist", "business", ["donor", "campaign", "case"], ["cultivate"], ["donor_retention"]),
    ("community_organizer_v1", "Community Organizer", "governance", ["community", "issue", "coalition"], ["mobilize"], ["collective_power"]),
    ("diplomat_v1", "Diplomat", "governance", ["actor", "interest", "agreement"], ["negotiate"], ["stability"]),
    ("public_relations_specialist_v1", "Public Relations Specialist", "business", ["public", "message", "reputation"], ["communicate"], ["trust"]),
    ("crisis_communications_manager_v1", "Crisis Communications Manager", "business", ["incident", "stakeholder", "message"], ["communicate"], ["reputation_recovery"]),
    ("investigator_v1", "Investigator", "governance", ["lead", "evidence", "timeline"], ["investigate"], ["case_integrity"]),
    ("detective_analyst_v1", "Detective Analyst", "governance", ["case", "suspect", "pattern"], ["analyze"], ["investigative_value"]),
    ("forensic_scientist_v1", "Forensic Scientist", "research", ["sample", "chain_of_custody", "result"], ["analyze"], ["forensic_validity"]),
    ("geospatial_analyst_v1", "Geospatial Analyst", "research", ["location", "layer", "movement"], ["map"], ["spatial_accuracy"]),
    ("military_planner_v1", "Military Planner", "operations", ["mission", "terrain", "force"], ["plan"], ["mission_effectiveness"]),
    ("aviation_operations_specialist_v1", "Aviation Operations Specialist", "operations", ["flight", "crew", "weather"], ["coordinate"], ["flight_safety"]),
    ("pilot_assistant_v1", "Pilot Assistant", "operations", ["aircraft", "checklist", "weather"], ["monitor"], ["aviation_safety"]),
    ("maritime_operations_specialist_v1", "Maritime Operations Specialist", "operations", ["vessel", "route", "port"], ["coordinate"], ["maritime_safety"]),
    ("agronomist_v1", "Agronomist", "research", ["crop", "soil", "yield"], ["assess"], ["agricultural_productivity"]),
    ("food_scientist_v1", "Food Scientist", "research", ["ingredient", "process", "safety"], ["test"], ["food_quality"]),
    ("veterinary_researcher_v1", "Veterinary Researcher", "research", ["species", "disease", "treatment"], ["study"], ["animal_health"]),
    ("sports_scientist_v1", "Sports Scientist", "research", ["athlete", "load", "performance"], ["measure"], ["performance_gain"]),
    ("strength_conditioning_coach_v1", "Strength Conditioning Coach", "education", ["athlete", "program", "recovery"], ["coach"], ["training_adaptation"]),
    ("personal_trainer_v1", "Personal Trainer", "education", ["client", "exercise", "habit"], ["coach"], ["fitness_progress"]),
    ("wellness_coach_v1", "Wellness Coach", "education", ["habit", "stress", "recovery"], ["coach"], ["wellbeing"]),
    ("life_coach_v1", "Life Coach", "education", ["goal", "belief", "routine"], ["coach"], ["goal_progress"]),
    ("chef_v1", "Chef", "creative", ["ingredient", "technique", "menu"], ["cook"], ["taste"]),
    ("restaurant_manager_v1", "Restaurant Manager", "operations", ["service", "staff", "inventory"], ["coordinate"], ["guest_experience"]),
    ("hospitality_manager_v1", "Hospitality Manager", "operations", ["guest", "service", "property"], ["manage_service"], ["guest_satisfaction"]),
    ("travel_planner_v1", "Travel Planner", "operations", ["traveler", "itinerary", "constraint"], ["plan"], ["trip_fit"]),
    ("accessibility_specialist_v1", "Accessibility Specialist", "governance", ["barrier", "accommodation", "standard"], ["audit"], ["accessibility"]),
    ("sustainability_manager_v1", "Sustainability Manager", "governance", ["emission", "resource", "stakeholder"], ["measure"], ["sustainability_impact"]),
    ("energy_policy_analyst_v1", "Energy Policy Analyst", "governance", ["grid", "market", "policy"], ["analyze_impact"], ["policy_feasibility"]),
    ("education_policy_analyst_v1", "Education Policy Analyst", "governance", ["student", "school", "intervention"], ["evaluate_policy"], ["equity"]),
    ("health_policy_analyst_v1", "Health Policy Analyst", "governance", ["population", "coverage", "cost"], ["evaluate_policy"], ["health_equity"]),
    ("transportation_planner_v1", "Transportation Planner", "governance", ["route", "mode", "demand"], ["model"], ["mobility"]),
    ("housing_policy_analyst_v1", "Housing Policy Analyst", "governance", ["housing_supply", "affordability", "tenant"], ["analyze_impact"], ["housing_access"]),
    ("labor_economist_v1", "Labor Economist", "research", ["worker", "wage", "market"], ["model"], ["labor_market_validity"]),
    ("demographer_v1", "Demographer", "research", ["population", "cohort", "migration"], ["forecast"], ["population_accuracy"]),
    ("survey_methodologist_v1", "Survey Methodologist", "research", ["questionnaire", "sample", "response"], ["design_survey"], ["measurement_quality"]),
    ("evaluation_specialist_v1", "Evaluation Specialist", "research", ["program", "outcome", "indicator"], ["evaluate"], ["impact_validity"]),
]

TARGET_PROFESSION_PRESET_COUNT = 216
PROFESSION_SPECS.extend(
    _MORE_PROFESSION_SPECS[: max(0, TARGET_PROFESSION_PRESET_COUNT - len(PROFESSION_SPECS))]
)

TRAIT_KEYWORDS = {
    "exploration": [
        "hypothesis", "idea", "novel", "new", "unknown", "experiment", "research",
        "假设", "新", "未知", "探索", "实验", "研究", "灵感",
    ],
    "rigor": [
        "evidence", "proof", "verify", "validated", "test", "metric", "source",
        "证据", "验证", "测试", "指标", "来源", "可靠", "复现",
    ],
    "conscientiousness": [
        "todo", "deadline", "step", "checklist", "must", "should", "process",
        "任务", "步骤", "清单", "必须", "应该", "流程", "规范",
    ],
    "risk_sensitivity": [
        "risk", "failure", "bug", "error", "unsafe", "red flag", "security",
        "风险", "失败", "错误", "故障", "安全", "红旗", "危险",
    ],
    "social_warmth": [
        "user", "customer", "patient", "preference", "emotion", "trust",
        "用户", "客户", "患者", "偏好", "情绪", "信任", "沟通",
    ],
    "abstraction_preference": [
        "architecture", "principle", "pattern", "model", "framework", "system",
        "架构", "原则", "模式", "模型", "框架", "系统", "抽象",
    ],
    "execution": [
        "implement", "fix", "deploy", "complete", "action", "plan", "next",
        "实现", "修复", "部署", "完成", "行动", "计划", "下一步",
    ],
}

CONCEPT_ALIASES = {
    "requirement": ["requirement", "need", "spec", "需求", "要求"],
    "constraint": ["constraint", "limit", "must", "cannot", "约束", "限制", "必须", "不能"],
    "interface": ["interface", "api", "contract", "endpoint", "接口", "契约"],
    "architecture": ["architecture", "module", "system", "design", "架构", "模块", "系统", "设计"],
    "tradeoff": ["tradeoff", "alternative", "option", "权衡", "取舍", "方案"],
    "failure_mode": ["failure", "bug", "regression", "incident", "失败", "故障", "错误", "回归"],
    "test": ["test", "verify", "validate", "coverage", "测试", "验证", "覆盖"],
    "deployment": ["deploy", "release", "上线", "部署", "发布"],
    "dependency": ["dependency", "package", "library", "依赖", "库"],
    "question": ["question", "problem", "问题"],
    "hypothesis": ["hypothesis", "assumption", "假设", "猜想"],
    "variable": ["variable", "control", "变量", "控制"],
    "method": ["method", "protocol", "方法", "方案"],
    "evidence": ["evidence", "data", "result", "证据", "数据", "结果"],
    "confounder": ["confounder", "bias", "混杂", "偏差"],
    "replication": ["replicate", "reproduce", "复现", "重复"],
    "uncertainty": ["uncertain", "confidence", "不确定", "置信"],
    "symptom": ["symptom", "症状"],
    "history": ["history", "病史", "历史"],
    "sign": ["sign", "体征"],
    "risk_factor": ["risk factor", "风险因素"],
    "differential_diagnosis": ["differential", "鉴别诊断"],
    "red_flag": ["red flag", "红旗", "警讯"],
    "contraindication": ["contraindication", "禁忌"],
    "follow_up": ["follow up", "随访", "复查"],
}

METHOD_ALIASES = {
    "clarify_requirements": ["clarify", "requirement", "澄清", "需求"],
    "decompose_system": ["decompose", "break down", "拆解", "分解"],
    "compare_alternatives": ["compare", "alternative", "tradeoff", "比较", "方案", "权衡"],
    "implement": ["implement", "code", "实现", "代码"],
    "debug": ["debug", "fix", "排查", "调试", "修复"],
    "review": ["review", "audit", "检查", "评审"],
    "refactor": ["refactor", "重构"],
    "verify": ["verify", "test", "验证", "测试"],
    "validate": ["validate", "验收", "确认"],
    "formulate": ["formulate", "提出", "构造"],
    "operationalize": ["operationalize", "操作化"],
    "experiment": ["experiment", "实验"],
    "analyze": ["analyze", "分析"],
    "replicate": ["replicate", "reproduce", "复现"],
    "compare_explanations": ["explanation", "解释", "对比"],
    "revise_theory": ["revise", "theory", "修正", "理论"],
    "collect_history": ["history", "采集", "病史"],
    "triage": ["triage", "分诊"],
    "differential": ["differential", "鉴别"],
    "risk_stratify": ["stratify", "risk", "分层", "风险"],
    "test_selection": ["test", "检查", "选择"],
    "safety_net": ["safety net", "安全网", "警示"],
}

EVALUATION_ALIASES = {
    "correctness": ["correct", "bug", "正确", "错误"],
    "maintainability": ["maintain", "readable", "维护", "可读"],
    "reliability": ["reliable", "stable", "稳定", "可靠"],
    "security": ["security", "安全"],
    "performance": ["performance", "latency", "性能", "延迟"],
    "simplicity": ["simple", "simplicity", "简单"],
    "test_coverage": ["coverage", "覆盖"],
    "validity": ["validity", "有效性"],
    "reproducibility": ["reproducible", "复现"],
    "effect_size": ["effect size", "效应"],
    "explanatory_power": ["explain", "解释力"],
    "novelty": ["novel", "新颖"],
    "falsifiability": ["falsify", "可证伪"],
    "patient_safety": ["patient safety", "患者安全"],
    "diagnostic_coverage": ["diagnostic coverage", "诊断覆盖"],
    "guideline_alignment": ["guideline", "指南"],
    "false_negative_risk": ["false negative", "漏诊"],
    "abstention_quality": ["abstain", "不确定", "转诊"],
}

ERROR_TERMS = [
    "failure", "failed", "bug", "error", "incident", "regression", "wrong",
    "mistake", "unsafe", "missed", "风险", "失败", "错误", "故障",
    "事故", "回归", "误判", "漏", "危险",
]

PROCEDURAL_TERMS = [
    "step", "checklist", "method", "process", "rule", "must", "should",
    "步骤", "清单", "方法", "流程", "规则", "必须", "应该",
]

EPISODIC_TERMS = [
    "happened", "conversation", "meeting", "today", "yesterday", "case",
    "发生", "会议", "今天", "昨天", "案例", "这次",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def parse_json_object(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    if not value:
        return deepcopy(default or {})
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return deepcopy(default or {})
        return parsed if isinstance(parsed, dict) else deepcopy(default or {})
    return deepcopy(default or {})


def parse_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return deepcopy(value)
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def list_personality_presets() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in PERSONALITY_PRESETS.values()]


def get_personality_preset(personality_id: str | None) -> dict[str, Any] | None:
    if not personality_id:
        return None
    preset = PERSONALITY_PRESETS.get(personality_id)
    return deepcopy(preset) if preset else None


def _build_profession_preset(spec: tuple) -> dict[str, Any]:
    profession_id, name, domain, extra_objects, extra_methods, extra_eval = spec
    base = PROFESSION_DOMAIN_PACKS[domain]
    objects = list(dict.fromkeys([*base["objects"], *extra_objects]))
    methods = list(dict.fromkeys([*base["methods"], *extra_methods]))
    evaluation = list(dict.fromkeys([*base["evaluation"], *extra_eval]))
    role = profession_id.removesuffix("_v1")
    object_phrase = ", ".join(objects[:8])
    method_phrase = ", ".join(methods[:8])
    evaluation_phrase = ", ".join(evaluation[:8])
    return {
        "profession_id": profession_id,
        "name": name,
        "description": (
            f"Professional cognition template for {name}: objects, methods, "
            "evaluation criteria, case memory, and error learning."
        ),
        "domain": domain,
        "profession": {
            "role": role,
            "objects": objects,
            "methods": methods,
            "evaluation": evaluation,
            "case_library_refs": list(base["case_refs"]),
            "error_library_refs": list(base["error_refs"]),
        },
        "development_plan": {
            "target_role": role,
            "stage": "foundation",
            "focus_areas": objects[:8],
            "promotion_rules": [
                "Promote repeated episodes into profession-specific semantic patterns.",
                "Promote repeated successful actions into procedural checklists.",
                "Promote failures, near misses, and ineffective recalls into an error library.",
            ],
        },
        "identity_prompt": (
            f"Act as a {name} memory lens. Treat {object_phrase} as the primary "
            "things worth noticing, naming, and preserving."
        ),
        "attention_prompt": [
            f"When reading new material, first ask which {name} object is present: {object_phrase}.",
            "Prefer cues that would change future professional judgment, not trivia.",
            "Mark uncertainty, missing evidence, and boundary conditions as first-class memory cues.",
        ],
        "method_prompt": [
            f"Use {name} methods such as {method_phrase}.",
            "Store method steps when they are reusable, repeatable, or prevent a known failure.",
            "Separate what happened, what was inferred, what was tried, and what changed.",
        ],
        "evaluation_prompt": [
            f"Judge usefulness with {evaluation_phrase}.",
            "Prefer memories that improve future decisions under the profession's evaluation criteria.",
            "Keep counterexamples when they prevent overgeneralization.",
        ],
        "error_prompt": [
            f"Build a {name} error library from failures, near misses, contradictions, and missed recalls.",
            "For each error memory, preserve trigger, mistaken assumption, consequence, and prevention rule.",
            "Raise retrieval priority for error memories when the current task resembles a prior failure.",
        ],
        "profession_prompt": [
            f"See the world as a {name}: prioritize {object_phrase}.",
            f"Use methods such as {method_phrase}.",
            f"Judge memory value by {evaluation_phrase}.",
            "Preserve both positive cases and error cases because expertise depends on contrast.",
        ],
    }


PROFESSION_PRESETS: dict[str, dict[str, Any]] = {
    spec[0]: _build_profession_preset(spec) for spec in PROFESSION_SPECS
}


def list_profession_presets() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in PROFESSION_PRESETS.values()]


def get_profession_preset(profession_id: str | None) -> dict[str, Any] | None:
    if not profession_id:
        return None
    preset = PROFESSION_PRESETS.get(profession_id)
    return deepcopy(preset) if preset else None


def compose_profile_id(personality_id: str, profession_id: str) -> str:
    personality_slug = personality_id.removesuffix("_v1")
    profession_slug = profession_id.removesuffix("_v1")
    return f"{personality_slug}__{profession_slug}_v1"


def compose_tendency_prompt(
    personality: dict[str, Any],
    profession_preset: dict[str, Any],
) -> dict[str, Any]:
    profession = profession_preset["profession"]
    personality_name = personality["name"]
    profession_name = profession_preset["name"]
    objects = profession.get("objects", [])
    methods = profession.get("methods", [])
    evaluation = profession.get("evaluation", [])
    return {
        "memory_identity_skeleton": deepcopy(MEMORY_IDENTITY_SKELETON),
        "system": [
            *MEMORY_IDENTITY_SKELETON["prompt"]["identity"],
            (
                f"Memory tendency is composed from personality '{personality_name}' "
                f"and profession '{profession_name}'. Personality shapes attention "
                "and execution pressure; profession shapes objects, methods, "
                "evaluation, cases, and errors."
            ),
            (
                "Do not treat the profession label as enough. The useful tendency "
                "emerges from the interaction between what this personality notices "
                "and what this profession considers important."
            ),
        ],
        "attention": [
            *personality.get("attention_prompt", []),
            profession_preset.get("identity_prompt", ""),
            *profession_preset.get("attention_prompt", []),
            *profession_preset.get("profession_prompt", [])[:1],
            (
                "When ingesting memory, ask: which professional object becomes "
                "salient because of this personality's attention style?"
            ),
        ],
        "execution": [
            *personality.get("execution_prompt", []),
            *profession_preset.get("method_prompt", []),
            f"Apply professional methods: {', '.join(methods[:10])}.",
            "Convert useful experience into repeatable professional action.",
        ],
        "storage": [
            *MEMORY_IDENTITY_SKELETON["prompt"]["write_policy"],
            *personality.get("memory_prompt", []),
            f"Prioritize professional objects: {', '.join(objects[:10])}.",
            f"Preserve evidence for evaluation criteria: {', '.join(evaluation[:10])}.",
            *profession_preset.get("evaluation_prompt", []),
            "Keep case memory, procedural memory, semantic abstractions, and error memory separate.",
        ],
        "retrieval": [
            *MEMORY_IDENTITY_SKELETON["prompt"]["retrieval_policy"],
            "Rank memories higher when they match both the current task and the composed tendency.",
            "Prefer memories whose stored professional cues align with the active question.",
            "Surface error-library memories when risk, failure, ambiguity, or missed recall is present.",
        ],
        "consolidation": [
            *MEMORY_IDENTITY_SKELETON["prompt"]["consolidation_policy"],
            "Merge repeated episodes into expert patterns only when they share professional cues.",
            "Distill repeated successful actions into procedural checklists.",
            "Distill repeated failures into prevention rules and diagnostic cues.",
        ],
        "error_library": profession_preset.get("error_prompt", []),
    }


def compose_profile_from_presets(
    personality_id: str,
    profession_id: str,
    profile_id: str | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    personality = get_personality_preset(personality_id)
    profession_preset = get_profession_preset(profession_id)
    if not personality:
        raise ValueError(f"Personality preset '{personality_id}' not found")
    if not profession_preset:
        raise ValueError(f"Profession preset '{profession_id}' not found")
    tendency_prompt = compose_tendency_prompt(personality, profession_preset)
    profile_id = profile_id or compose_profile_id(personality_id, profession_id)
    profile_name = name or f"{personality['name']} + {profession_preset['name']}"
    plan = deepcopy(profession_preset["development_plan"])
    plan["personality_id"] = personality_id
    plan["profession_id"] = profession_id
    plan["tendency_prompt"] = tendency_prompt
    return normalize_profile_payload(
        profile_id=profile_id,
        traits=personality.get("traits", {}),
        profession=profession_preset.get("profession", {}),
        development_plan=plan,
        calibration=DEFAULT_CALIBRATION,
        name=profile_name,
        description=(
            f"Composed memory profile from personality '{personality['name']}' "
            f"and profession '{profession_preset['name']}'."
        ),
        version="1",
    )


def refresh_profile_scaffold(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    """Attach the latest composed prompt scaffold while preserving user tuning."""
    if not profile:
        return None
    plan = parse_json_object(profile.get("development_plan"))
    personality_id = plan.get("personality_id")
    profession_id = plan.get("profession_id")
    if not personality_id or not profession_id:
        return profile
    try:
        fresh = compose_profile_from_presets(
            personality_id=personality_id,
            profession_id=profession_id,
            profile_id=profile.get("profile_id") or profile.get("id"),
            name=profile.get("name"),
        )
    except ValueError:
        return profile
    refreshed = deepcopy(profile)
    refreshed["profession"] = fresh["profession"]
    refreshed_plan = fresh["development_plan"]
    for key, value in plan.items():
        if key not in {"focus_areas", "promotion_rules", "tendency_prompt"}:
            refreshed_plan[key] = value
    refreshed["development_plan"] = refreshed_plan
    refreshed["calibration"] = {**fresh["calibration"], **parse_json_object(profile.get("calibration"))}
    refreshed["description"] = profile.get("description") or fresh["description"]
    return refreshed


def list_default_profiles() -> list[dict[str, Any]]:
    return [deepcopy(profile) for profile in DEFAULT_PROFILE_PACKS.values()]


def get_default_profile(profile_id: str) -> dict[str, Any] | None:
    profile = DEFAULT_PROFILE_PACKS.get(profile_id)
    return deepcopy(profile) if profile else None


def normalize_profile_payload(
    profile_id: str,
    traits: Any,
    profession: Any,
    development_plan: Any | None = None,
    calibration: Any | None = None,
    name: str | None = None,
    description: str | None = None,
    version: str = "1",
) -> dict[str, Any]:
    trait_obj = parse_json_object(traits)
    normalized_traits = {
        key: clamp(float(trait_obj.get(key, 0.5) or 0.0))
        for key in TRAIT_KEYS
    }
    profession_obj = parse_json_object(profession)
    normalized_profession = {
        "role": str(profession_obj.get("role") or profession_obj.get("name") or "custom"),
        "objects": [str(item) for item in profession_obj.get("objects", [])],
        "methods": [str(item) for item in profession_obj.get("methods", [])],
        "evaluation": [str(item) for item in profession_obj.get("evaluation", [])],
        "case_library_refs": [
            str(item) for item in profession_obj.get("case_library_refs", [])
        ],
        "error_library_refs": [
            str(item) for item in profession_obj.get("error_library_refs", [])
        ],
    }
    plan_obj = parse_json_object(development_plan)
    calibration_obj = {**DEFAULT_CALIBRATION, **parse_json_object(calibration)}
    return {
        "profile_id": profile_id,
        "name": name or profile_id,
        "description": description or "",
        "version": version,
        "traits": normalized_traits,
        "profession": normalized_profession,
        "development_plan": plan_obj,
        "calibration": calibration_obj,
    }


def build_runtime_profile(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "profile_id": row.get("profile_id") or row.get("id"),
        "name": row.get("name", ""),
        "description": row.get("description", ""),
        "version": row.get("version", "1"),
        "traits": parse_json_object(row.get("traits")),
        "profession": parse_json_object(row.get("profession")),
        "development_plan": parse_json_object(row.get("development_plan")),
        "calibration": {**DEFAULT_CALIBRATION, **parse_json_object(row.get("calibration"))},
        "source": row.get("source", ""),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term and term.lower() in text for term in terms)


def _hit_ratio(text: str, terms: list[str]) -> tuple[float, list[str]]:
    if not terms:
        return 0.0, []
    hits = []
    for term in terms:
        term_text = str(term).strip()
        if not term_text:
            continue
        aliases = [term_text, *CONCEPT_ALIASES.get(term_text, [])]
        if _contains_any(text, aliases):
            hits.append(term_text)
    denominator = max(1, min(len(terms), 8))
    return clamp(len(set(hits)) / denominator), sorted(set(hits))


def _method_hit_ratio(text: str, methods: list[str]) -> tuple[float, list[str]]:
    if not methods:
        return 0.0, []
    hits = []
    for method in methods:
        method_text = str(method).strip()
        aliases = [method_text, *METHOD_ALIASES.get(method_text, [])]
        if _contains_any(text, aliases):
            hits.append(method_text)
    denominator = max(1, min(len(methods), 8))
    return clamp(len(set(hits)) / denominator), sorted(set(hits))


def _evaluation_hit_ratio(text: str, metrics: list[str]) -> tuple[float, list[str]]:
    if not metrics:
        return 0.0, []
    hits = []
    for metric in metrics:
        metric_text = str(metric).strip()
        aliases = [metric_text, *EVALUATION_ALIASES.get(metric_text, [])]
        if _contains_any(text, aliases):
            hits.append(metric_text)
    denominator = max(1, min(len(metrics), 8))
    return clamp(len(set(hits)) / denominator), sorted(set(hits))


def infer_trait_features(text: str, memory_type: str | None = None) -> dict[str, float]:
    lowered = (text or "").lower()
    features = {}
    for key, terms in TRAIT_KEYWORDS.items():
        hits = sum(1 for term in terms if term.lower() in lowered)
        features[key] = clamp(hits / 3.0)

    memory_type = (memory_type or "").lower()
    if memory_type in {"architecture", "decision", "principle", "api_contract", "db_schema"}:
        features["abstraction_preference"] = max(features["abstraction_preference"], 0.75)
        features["rigor"] = max(features["rigor"], 0.55)
    if memory_type in {"bug", "error_pattern"}:
        features["risk_sensitivity"] = max(features["risk_sensitivity"], 0.85)
    if memory_type in {"task_snapshot", "plan", "deployment"}:
        features["execution"] = max(features["execution"], 0.75)
        features["conscientiousness"] = max(features["conscientiousness"], 0.65)
    if memory_type in {"preference"}:
        features["social_warmth"] = max(features["social_warmth"], 0.65)
    return features


def infer_profession_features(
    text: str,
    professional_profile: dict[str, Any],
    development_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lowered = (text or "").lower()
    objects = [str(item) for item in professional_profile.get("objects", [])]
    methods = [str(item) for item in professional_profile.get("methods", [])]
    evaluation = [str(item) for item in professional_profile.get("evaluation", [])]
    object_score, object_hits = _hit_ratio(lowered, objects)
    method_score, method_hits = _method_hit_ratio(lowered, methods)
    evaluation_score, evaluation_hits = _evaluation_hit_ratio(lowered, evaluation)

    plan = development_plan or {}
    focus_terms = [str(item) for item in plan.get("focus_areas", [])]
    focus_score, focus_hits = _hit_ratio(lowered, focus_terms)
    error_signal = _contains_any(lowered, ERROR_TERMS)
    procedural_signal = _contains_any(lowered, PROCEDURAL_TERMS)
    episodic_signal = _contains_any(lowered, EPISODIC_TERMS)

    relevance = clamp(
        object_score * 0.42
        + method_score * 0.25
        + evaluation_score * 0.20
        + focus_score * 0.13
    )
    learning_value = clamp(
        (0.35 if object_hits else 0.0)
        + (0.25 if method_hits else 0.0)
        + (0.25 if error_signal else 0.0)
        + (0.15 if focus_hits else 0.0)
    )
    return {
        "role": professional_profile.get("role", "custom"),
        "object_score": round(object_score, 4),
        "method_score": round(method_score, 4),
        "evaluation_score": round(evaluation_score, 4),
        "focus_score": round(focus_score, 4),
        "professional_relevance": round(relevance, 4),
        "learning_value": round(learning_value, 4),
        "object_hits": object_hits,
        "method_hits": method_hits,
        "evaluation_hits": evaluation_hits,
        "focus_hits": focus_hits,
        "error_signal": error_signal,
        "procedural_signal": procedural_signal,
        "episodic_signal": episodic_signal,
    }


def trait_affinity(trait_features: dict[str, float], traits: dict[str, Any]) -> float:
    numerator = 0.0
    denominator = 0.0
    for key in TRAIT_KEYS:
        profile_value = clamp(float(traits.get(key, 0.5) or 0.0))
        feature_value = clamp(float(trait_features.get(key, 0.0) or 0.0))
        numerator += profile_value * feature_value
        denominator += profile_value
    if denominator <= 0:
        return 0.0
    return clamp(numerator / denominator)


def trait_mismatch(trait_features: dict[str, float], traits: dict[str, Any]) -> float:
    """Small penalty when the active profile is strong but content has no matching cues."""
    high_profile_keys = [key for key in TRAIT_KEYS if float(traits.get(key, 0.0) or 0.0) >= 0.75]
    if not high_profile_keys:
        return 0.0
    missing = [
        key for key in high_profile_keys
        if float(trait_features.get(key, 0.0) or 0.0) < 0.15
    ]
    return clamp(len(missing) / max(1, len(high_profile_keys)) * 0.25)


def route_memory_layer(
    memory_type: str | None,
    profession_features: dict[str, Any],
    source_type: str | None = None,
) -> str:
    memory_type = (memory_type or "").lower()
    source_type = (source_type or "").lower()
    if memory_type in {"bug", "error_pattern"} or profession_features.get("error_signal"):
        return "error"
    if memory_type in {"coding_rule", "product_rule", "principle", "method", "checklist"}:
        return "procedural"
    if profession_features.get("procedural_signal"):
        return "procedural"
    if memory_type in {"event", "task_snapshot"} or source_type in {
        "conversation", "update", "execution", "case",
    } or profession_features.get("episodic_signal"):
        return "episodic"
    return "semantic"


def risk_penalty(text: str, profile: dict[str, Any], profession_features: dict[str, Any]) -> float:
    profession = profile.get("profession", {})
    role = str(profession.get("role", "")).lower()
    lowered = (text or "").lower()
    if "clinical" not in role and "doctor" not in role:
        return 0.0
    unsupported_action = _contains_any(
        lowered,
        ["diagnose", "prescribe", "treatment", "确诊", "开药", "治疗方案"],
    )
    has_uncertainty = _contains_any(
        lowered,
        ["uncertain", "maybe", "risk", "follow up", "不确定", "风险", "随访", "转诊"],
    )
    if unsupported_action and not has_uncertainty:
        return 0.65
    if profession_features.get("error_signal"):
        return 0.0
    return 0.0


def score_memory_with_profile(
    text: str,
    memory_type: str,
    base_score: float,
    profile: dict[str, Any] | None,
    source_type: str | None = None,
) -> dict[str, Any]:
    base_score = clamp(float(base_score or 0.0))
    if not profile:
        return {
            "profile_id": None,
            "base_score": round(base_score, 4),
            "store_score": round(base_score, 4),
            "memory_layer": route_memory_layer(memory_type, {}, source_type),
            "trait_features": {},
            "profession_features": {},
            "score_reason": "No active memory profile; used base score.",
        }

    traits = parse_json_object(profile.get("traits"))
    profession = parse_json_object(profile.get("profession"))
    development_plan = parse_json_object(profile.get("development_plan"))
    calibration = {**DEFAULT_CALIBRATION, **parse_json_object(profile.get("calibration"))}

    trait_features = infer_trait_features(text, memory_type)
    prof_features = infer_profession_features(text, profession, development_plan)
    affinity = trait_affinity(trait_features, traits)
    mismatch = trait_mismatch(trait_features, traits)
    profession_relevance = float(prof_features.get("professional_relevance", 0.0) or 0.0)
    learning_value = float(prof_features.get("learning_value", 0.0) or 0.0)
    penalty = risk_penalty(text, profile, prof_features)

    store_score = clamp(
        base_score
        + calibration["trait_weight"] * affinity
        - calibration["trait_weight"] * mismatch
        + calibration["profession_weight"] * profession_relevance
        + calibration["learning_weight"] * learning_value
        - calibration["risk_penalty_weight"] * penalty
    )
    layer = route_memory_layer(memory_type, prof_features, source_type)
    reason = (
        f"base={base_score:.3f}; trait_affinity={affinity:.3f}; "
        f"trait_mismatch={mismatch:.3f}; profession={profession_relevance:.3f}; "
        f"learning={learning_value:.3f}; risk_penalty={penalty:.3f}; layer={layer}"
    )
    return {
        "profile_id": profile.get("profile_id") or profile.get("id"),
        "base_score": round(base_score, 4),
        "store_score": round(store_score, 4),
        "memory_layer": layer,
        "trait_features": {key: round(value, 4) for key, value in trait_features.items()},
        "profession_features": prof_features,
        "score_reason": reason,
    }


def score_retrieval_with_profile(
    memory: dict[str, Any],
    task: str,
    profile: dict[str, Any] | None,
    semantic_relevance: float | None = None,
) -> dict[str, Any]:
    if not profile:
        base = float(memory.get("semantic_similarity", semantic_relevance or 0.0) or 0.0)
        if base <= 0:
            base = float(memory.get("score", 0.0) or 0.0)
        return {"retrieve_score": round(clamp(base), 4), "retrieve_reason": "No active profile."}

    task_text = task or ""
    memory_text = f"{memory.get('title', '')} {memory.get('content', '')}"
    profession = parse_json_object(profile.get("profession"))
    plan = parse_json_object(profile.get("development_plan"))
    task_prof = infer_profession_features(task_text, profession, plan)
    memory_prof = parse_json_object(memory.get("profession_features"))
    if not memory_prof:
        memory_prof = infer_profession_features(memory_text, profession, plan)

    semantic = semantic_relevance
    if semantic is None:
        semantic = float(memory.get("semantic_similarity", 0.0) or 0.0)
    if semantic <= 0:
        semantic = float(memory.get("score", 0.0) or 0.0)

    task_hits = set(task_prof.get("object_hits", [])) | set(task_prof.get("method_hits", []))
    memory_hits = set(memory_prof.get("object_hits", [])) | set(memory_prof.get("method_hits", []))
    shared = len(task_hits & memory_hits)
    cue_alignment = shared / math.sqrt(max(1, len(task_hits)) * max(1, len(memory_hits)))
    evidence_quality = float(memory.get("confidence_score", 0.5) or 0.5)
    error_utility = 1.0 if (
        memory.get("memory_layer") == "error"
        and (task_prof.get("error_signal") or "risk" in task_text.lower() or "风险" in task_text)
    ) else 0.0
    recency_usefulness = 0.15 if memory.get("updated_at") else 0.0
    retrieve_score = clamp(
        0.45 * clamp(float(semantic or 0.0))
        + 0.25 * clamp(cue_alignment)
        + 0.15 * clamp(evidence_quality)
        + 0.10 * error_utility
        + 0.05 * recency_usefulness
    )
    reason = (
        f"semantic={float(semantic or 0.0):.3f}; cue_alignment={cue_alignment:.3f}; "
        f"evidence={evidence_quality:.3f}; error_utility={error_utility:.3f}"
    )
    return {"retrieve_score": round(retrieve_score, 4), "retrieve_reason": reason}
