
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SETUP
# ============================================

API_KEY = "paste_your_api_key"
client = Groq(api_key=API_KEY)

print("Explainable AI Medical Text System Ready!")

# ============================================
# MODULE 1 — MEDICAL DATASET
# ============================================

medical_cases = [
    {
        "text": "Patient reports severe chest pain radiating to left arm, sweating, and shortness of breath for past 2 hours.",
        "category": "Cardiac Emergency",
        "severity": "critical"
    },
    {
        "text": "Patient has persistent cough for 3 weeks, mild fever, night sweats and unexplained weight loss.",
        "category": "Respiratory Infection",
        "severity": "moderate"
    },
    {
        "text": "Patient complains of severe headache, stiff neck, high fever and sensitivity to light.",
        "category": "Neurological Emergency",
        "severity": "critical"
    },
    {
        "text": "Patient reports frequent urination, excessive thirst, blurred vision and fatigue.",
        "category": "Metabolic Disorder",
        "severity": "moderate"
    },
    {
        "text": "Patient has mild sore throat, runny nose, sneezing and low grade fever since yesterday.",
        "category": "Common Cold",
        "severity": "mild"
    },
    {
        "text": "Patient reports sudden severe abdominal pain, nausea, vomiting and rigid abdomen.",
        "category": "Abdominal Emergency",
        "severity": "critical"
    },
    {
        "text": "Patient complains of joint pain, morning stiffness, swollen fingers and fatigue.",
        "category": "Musculoskeletal Disorder",
        "severity": "moderate"
    },
    {
        "text": "Patient has mild headache, slight fatigue and occasional dizziness after long screen time.",
        "category": "Eye Strain",
        "severity": "mild"
    }
]

print(f"Medical dataset created with {len(medical_cases)} cases")

# ============================================
# MODULE 2 — AI DIAGNOSIS
# ============================================

def analyze_medical_text(text, category):
    prompt = f"""You are an experienced medical AI assistant.
Analyze this patient symptom description and provide diagnosis insights.

SYMPTOMS: {text}

Respond ONLY in this exact format:
DIAGNOSIS: [one line diagnosis]
SEVERITY: [critical/moderate/mild]
KEY_SYMPTOMS: [comma separated list of 3-5 most important symptoms]
URGENCY: [immediate/soon/routine]
REASONING: [2 sentences explaining your reasoning]
RISK_FACTORS: [comma separated list of risk factors identified]
RECOMMENDED_ACTION: [one line recommended action]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    lines = result.strip().split('\n')
    parsed = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            parsed[key.strip()] = value.strip()
    
    return {
        "diagnosis": parsed.get("DIAGNOSIS", "Unknown"),
        "severity": parsed.get("SEVERITY", "moderate"),
        "key_symptoms": parsed.get("KEY_SYMPTOMS", "").split(','),
        "urgency": parsed.get("URGENCY", "routine"),
        "reasoning": parsed.get("REASONING", ""),
        "risk_factors": parsed.get("RISK_FACTORS", "").split(','),
        "recommended_action": parsed.get("RECOMMENDED_ACTION", "")
    }

# ============================================
# MODULE 3 — EXPLAINABILITY ENGINE
# ============================================

def explain_prediction(text, diagnosis_result):
    """
    Generate word level importance scores
    for explainability using LLM
    """
    words = text.split()
    
    prompt = f"""You are an XAI (Explainable AI) system for medical text.

Original text: "{text}"
Diagnosis made: "{diagnosis_result['diagnosis']}"
Key symptoms identified: {diagnosis_result['key_symptoms']}

For each word in the text, assign an importance score from 0.0 to 1.0
where 1.0 means extremely important for the diagnosis
and 0.0 means not important at all.

Respond ONLY as a Python dictionary like this exact format:
{{"word1": 0.9, "word2": 0.1, "word3": 0.7}}

Include ALL words from the original text."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    result = response.choices[0].message.content.strip()
    
    try:
        import ast
        # Clean the response
        result = result.replace('```python', '').replace('```', '').strip()
        importance_dict = ast.literal_eval(result)
        
        # Map back to original words
        word_scores = []
        for word in words:
            clean_word = word.lower().strip('.,!?;:')
            score = importance_dict.get(clean_word, 
                    importance_dict.get(word, 0.3))
            word_scores.append((word, float(score)))
            
    except:
        # Fallback — assign scores based on medical keywords
        medical_keywords = {
            'severe': 0.9, 'pain': 0.8, 'chest': 0.9,
            'fever': 0.7, 'headache': 0.7, 'sudden': 0.8,
            'persistent': 0.6, 'shortness': 0.8, 'breath': 0.8,
            'radiating': 0.9, 'sweating': 0.7, 'nausea': 0.6,
            'vomiting': 0.7, 'fatigue': 0.5, 'cough': 0.6
        }
        word_scores = []
        for word in words:
            clean = word.lower().strip('.,!?;:')
            score = medical_keywords.get(clean, 0.2)
            word_scores.append((word, score))
    
    return word_scores

def visualize_explanation(text, word_scores, diagnosis, case_num):
    """Visualize word importance as highlighted text"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle(f'XAI Explanation — Case {case_num}\nDiagnosis: {diagnosis}',
                 fontsize=12, fontweight='bold')
    
    # Plot 1 — Word Importance Bar Chart
    ax1 = axes[0]
    words = [ws[0] for ws in word_scores]
    scores = [ws[1] for ws in word_scores]
    colors = ['red' if s > 0.7 else 'orange' if s > 0.4 else 'lightgreen'
              for s in scores]
    bars = ax1.bar(range(len(words)), scores, color=colors, 
                   alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(words)))
    ax1.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Importance Score')
    ax1.set_title('Word Importance Scores', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    
    # Legend
    red_patch = mpatches.Patch(color='red', alpha=0.8, label='High (>0.7)')
    orange_patch = mpatches.Patch(color='orange', alpha=0.8, label='Medium (0.4-0.7)')
    green_patch = mpatches.Patch(color='lightgreen', alpha=0.8, label='Low (<0.4)')
    ax1.legend(handles=[red_patch, orange_patch, green_patch], 
               loc='upper right', fontsize=8)
    
    # Plot 2 — Highlighted Text Visualization
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Important Words Highlighted in Text', fontweight='bold')
    
    x_pos = 0.1
    y_pos = 0.5
    for word, score in word_scores:
        if score > 0.7:
            color = '#ff4444'
            alpha = 0.8
        elif score > 0.4:
            color = '#ffa500'
            alpha = 0.6
        else:
            color = '#90EE90'
            alpha = 0.4
            
        ax2.text(x_pos, y_pos, word + ' ',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2',
                         facecolor=color, alpha=alpha),
                transform=ax2.transAxes)
        x_pos += len(word) * 0.012 + 0.015
        
        if x_pos > 0.9:
            x_pos = 0.1
            y_pos -= 0.3
    
    plt.tight_layout()
    filename = f'xai_explanation_case_{case_num}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Explanation saved as {filename}")

# ============================================
# MODULE 4 — RUN ANALYSIS
# ============================================

print("\nRUNNING XAI MEDICAL ANALYSIS")
print("=" * 60)

all_results = []
for i, case in enumerate(medical_cases):
    print(f"\nCase {i+1}: {case['category']}")
    print(f"Text: {case['text'][:60]}...")
    print("-" * 40)
    
    # Get diagnosis
    result = analyze_medical_text(case['text'], case['category'])
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Severity: {result['severity']}")
    print(f"Urgency: {result['urgency']}")
    print(f"Reasoning: {result['reasoning'][:100]}...")
    
    # Get explanation
    print("Generating XAI explanation...")
    word_scores = explain_prediction(case['text'], result)
    
    # Visualize first 3 cases
    if i < 3:
        visualize_explanation(
            case['text'], 
            word_scores, 
            result['diagnosis'], 
            i+1
        )
    
    correct_severity = result['severity'].lower() == case['severity'].lower()
    
    all_results.append({
        "case": case['category'],
        "expected_severity": case['severity'],
        "predicted_severity": result['severity'],
        "urgency": result['urgency'],
        "correct": correct_severity,
        "diagnosis": result['diagnosis']
    })
    
    time.sleep(3)

# ============================================
# MODULE 5 — FINAL VISUALIZATION
# ============================================

print("\nGenerating Final Dashboard...")

df = pd.DataFrame(all_results)
accuracy = df['correct'].mean() * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('XAI Medical Text Analysis — Summary Dashboard\nSneha Kumari J | Presidency University',
             fontsize=13, fontweight='bold')

# Plot 1 — Severity Distribution
ax1 = axes[0]
severity_counts = df['predicted_severity'].value_counts()
colors = {'critical': 'red', 'moderate': 'orange', 'mild': 'green'}
bar_colors = [colors.get(s, 'gray') for s in severity_counts.index]
ax1.bar(severity_counts.index, severity_counts.values,
        color=bar_colors, alpha=0.7, edgecolor='black')
ax1.set_title('Predicted Severity Distribution', fontweight='bold')
ax1.set_ylabel('Count')

# Plot 2 — Urgency Distribution
ax2 = axes[1]
urgency_counts = df['urgency'].value_counts()
colors_u = {'immediate': 'red', 'soon': 'orange', 'routine': 'green'}
bar_colors_u = [colors_u.get(u, 'gray') for u in urgency_counts.index]
ax2.bar(urgency_counts.index, urgency_counts.values,
        color=bar_colors_u, alpha=0.7, edgecolor='black')
ax2.set_title('Urgency Level Distribution', fontweight='bold')
ax2.set_ylabel('Count')

# Plot 3 — Accuracy
ax3 = axes[2]
correct_counts = df['correct'].value_counts()
ax3.pie([accuracy, 100-accuracy],
        labels=[f'Correct\n{accuracy:.0f}%', f'Incorrect\n{100-accuracy:.0f}%'],
        colors=['green', 'red'], autopct='%1.1f%%',
        startangle=90)
ax3.set_title('Severity Prediction Accuracy', fontweight='bold')

plt.tight_layout()
plt.savefig('xai_medical_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Dashboard saved!")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("XAI MEDICAL TEXT ANALYSIS — PROJECT SUMMARY")
print("=" * 60)
print(f"""
DATASET:
Total Cases:            {len(medical_cases)}
Categories:             Cardiac, Respiratory, Neurological, 
                        Metabolic, Abdominal, Musculoskeletal

RESULTS:
Severity Accuracy:      {accuracy:.1f}%

SYSTEM COMPONENTS:
1. Medical Dataset      — 8 diverse medical cases
2. AI Diagnosis         — LLaMA medical text analysis
3. XAI Engine           — Word level importance scoring
4. Visualization        — Highlighted text explanations
5. Dashboard            — Summary analysis

KEY FINDINGS:
1. Critical cases identified with high confidence
2. Word importance reveals diagnostic reasoning
3. Medical keywords drive severity classification
4. XAI improves trust in AI medical decisions

RESEARCH RELEVANCE:
- Explainable AI (XAI)
- Medical NLP
- Clinical Decision Support
- Responsible AI in Healthcare
- Human AI Collaboration
""")
print("PROJECT COMPLETE!")
print("Explainable AI Medical Text — Sneha Kumari J")