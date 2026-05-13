"""
Diabetes Risk Predictor — CLI natural language interface.
Run with: python src/app.py
"""
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import mlflow.sklearn

load_dotenv()

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import FEATURES  # noqa: E402

# ── Feature definitions ───────────────────────────────────────────────────────

KEY_MAP = {
    'pregnancies':                'Pregnancies',
    'glucose':                    'Glucose',
    'blood_pressure':             'BloodPressure',
    'skin_thickness':             'SkinThickness',
    'insulin':                    'Insulin',
    'bmi':                        'BMI',
    'diabetes_pedigree_function': 'DiabetesPedigreeFunction',
    'age':                        'Age',
}

LABELS = {
    'pregnancies':                'Pregnancies',
    'glucose':                    'Glucose (mg/dL)',
    'blood_pressure':             'Diastolic BP (mm Hg)',
    'skin_thickness':             'Skin Thickness (mm)',
    'insulin':                    'Insulin (mu U/ml)',
    'bmi':                        'BMI',
    'diabetes_pedigree_function': 'Diabetes Pedigree Function',
    'age':                        'Age (years)',
}

MODEL = 'meta-llama/Llama-3.3-70B-Instruct'

PARSE_SYSTEM = """\
Extract clinical features for a diabetes risk model from the user's text.

Return ONLY a JSON object with these keys (use null if not mentioned):
pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, out_of_scope

Set out_of_scope to true if the input is a general question, greeting, or anything unrelated to a patient's
clinical information (e.g. "what is diabetes", "hello", "what does BMI mean").
Otherwise set out_of_scope to false.
"""

EXPLAIN_SYSTEM = """\
You are a helpful health assistant. Explain a diabetes risk model's output in simple, supportive terms.
Always remind the user that this is an ML model and they should consult a doctor for real medical decisions.
"""

# ── Core functions ────────────────────────────────────────────────────────────

def parse_features(client, text: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': PARSE_SYSTEM},
            {'role': 'user', 'content': text},
        ],
        temperature=0,
        max_tokens=200,
    )
    raw = response.choices[0].message.content.strip()
    if '```' in raw:
        raw = raw.split('```')[1].lstrip('json').strip()
    return json.loads(raw)


def run_inference(pipeline, features: dict) -> tuple[int, float]:
    row = {KEY_MAP[k]: (v if v is not None else np.nan)
           for k, v in features.items() if k in KEY_MAP}
    X = pd.DataFrame([row])
    prob = float(pipeline.predict_proba(X)[0, 1])
    return int(prob >= 0.5), prob


def explain_prediction(client, features: dict, pred: int, prob: float) -> str:
    known = '\n'.join(f'- {LABELS[k]}: {v}'
                     for k, v in features.items() if k in LABELS and v is not None)
    label = 'HIGH risk of diabetes' if pred == 1 else 'LOW risk of diabetes'
    prompt = (
        f'Patient info:\n{known}\n\n'
        f'Model result: {label} ({prob:.1%} probability)\n\n'
        'Explain this result in 3-4 sentences.'
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {'role': 'system', 'content': EXPLAIN_SYSTEM},
            {'role': 'user', 'content': prompt},
        ],
        temperature=0.3,
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def load_model():
    with open(PROJECT_ROOT / 'configs' / 'best_model.json') as f:
        info = json.load(f)
    mlflow.set_tracking_uri(info['tracking_uri'])
    pipeline = mlflow.sklearn.load_model(f"runs:/{info['run_id']}/model")
    return pipeline, info


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print('Loading model...')
    pipeline, model_info = load_model()
    print(f"Model: {model_info['model_name']} | ROC-AUC: {model_info['metrics']['roc_auc']:.4f}")

    api_key = os.getenv('NEBIUS_API_KEY')
    client = None
    if api_key:
        client = OpenAI(base_url='https://api.studio.nebius.com/v1/', api_key=api_key)
        print('LLM mode: active (Nebius API key detected)')
    else:
        print('LLM mode: disabled (no NEBIUS_API_KEY found — add it to your .env file)')

    print('\nDiabetes Risk Predictor — type "quit" to exit')
    print('Example: "45-year-old woman, BMI 32, fasting glucose 148, 2 pregnancies"\n')

    while True:
        user_input = input('You: ').strip()
        if user_input.lower() in ('quit', 'exit'):
            break
        if not user_input:
            continue

        if client is None:
            print('Assistant: LLM is not available. Please set NEBIUS_API_KEY in your .env file.\n')
            continue

        try:
            features = parse_features(client, user_input)
        except Exception as e:
            print(f'Assistant: Could not parse input ({e}). Please try rephrasing.\n')
            continue

        if features.get('out_of_scope'):
            print('Assistant: I can only assess diabetes risk from clinical information. '
                  'Please describe a patient — for example their age, BMI, glucose level, '
                  'and number of pregnancies.\n')
            continue

        missing = [LABELS[k] for k, v in features.items()
                   if k != 'out_of_scope' and v is None]

        # Ask for clarification if fewer than 3 features were provided
        if len(missing) > 5:
            print(f'Assistant: I need more information to make a reliable prediction. '
                  f'Could you provide some of these: {", ".join(missing)}?\n')
            continue

        if missing:
            print(f'Note: {", ".join(missing)} not mentioned — model will use median values.')

        pred, prob = run_inference(pipeline, features)
        label = 'High Risk' if pred == 1 else 'Low Risk'
        explanation = explain_prediction(client, features, pred, prob)

        print(f'\nRisk Level : {label} ({prob:.1%} probability of diabetes)')
        print(f'Assistant  : {explanation}\n')


if __name__ == '__main__':
    main()
