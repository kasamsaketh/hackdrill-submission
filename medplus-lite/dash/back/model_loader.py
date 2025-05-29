from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import logging
import re
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables
tokenizer = None
model = None
qa_pipeline = None

def initialize_model():
    global tokenizer, model, qa_pipeline
    
    try:
        logger.info("Loading conversational model...")
        
        # Try using better models for medical/general knowledge
        model_options = [
            "microsoft/DialoGPT-medium",
            "distilgpt2",
            "gpt2"
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load {model_name}...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                logger.info(f"Successfully loaded {model_name} on {device}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if model is None:
            raise Exception("All model loading attempts failed")
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        tokenizer = None
        model = None

# Initialize model on import
initialize_model()

def get_comprehensive_medical_answer(question):
    """Expanded medical knowledge base covering more topics"""
    question_lower = question.lower()
    
    medical_knowledge = {
        # Infections and Antibiotics
        'bacterial_infections': {
            'keywords': ['antibiotics', 'bacterial', 'meningitis', 'pneumonia', 'sepsis', 'strep throat', 'uti', 'urinary tract infection'],
            'answer': "Bacterial infections require antibiotic treatment. Common bacterial infections include pneumonia, urinary tract infections, strep throat, and bacterial meningitis. The specific antibiotic depends on the type of bacteria and infection severity. Always complete the full antibiotic course as prescribed, even if symptoms improve."
        },
        
        # Pain Management
        'pain_management': {
            'keywords': ['pain', 'ache', 'sore', 'hurt', 'painful', 'chronic pain', 'acute pain'],
            'answer': "Pain management varies by type and severity. For mild to moderate pain, over-the-counter medications like acetaminophen, ibuprofen, or aspirin can help. Other methods include rest, ice/heat therapy, gentle exercise, and stress management. Chronic pain may require prescription medications, physical therapy, or specialized treatments. Always consult a healthcare provider for persistent or severe pain."
        },
        
        # Respiratory Issues
        'respiratory': {
            'keywords': ['cough', 'breathing', 'shortness of breath', 'asthma', 'bronchitis', 'chest congestion', 'wheezing'],
            'answer': "Respiratory symptoms like cough, shortness of breath, or wheezing can have various causes including infections, asthma, allergies, or more serious conditions. Treatment depends on the underlying cause. For mild symptoms, rest, fluids, and humidified air may help. Seek immediate medical attention for severe breathing difficulties, chest pain, or persistent symptoms."
        },
        
        # Digestive Issues
        'digestive': {
            'keywords': ['stomach', 'nausea', 'vomiting', 'diarrhea', 'constipation', 'indigestion', 'heartburn', 'acid reflux'],
            'answer': "Digestive issues can range from mild to serious. For nausea and vomiting, try clear fluids, bland foods, and rest. Diarrhea requires fluid replacement to prevent dehydration. Constipation may improve with fiber, fluids, and exercise. Heartburn can be managed with antacids and dietary changes. Seek medical care for severe symptoms, blood in stool/vomit, or persistent issues."
        },
        
        # Skin Conditions
        'skin_conditions': {
            'keywords': ['rash', 'eczema', 'dermatitis', 'acne', 'skin irritation', 'itching', 'dry skin', 'skin infection'],
            'answer': "Skin conditions have various causes including allergies, infections, or chronic conditions like eczema. Treatment depends on the specific condition. General care includes gentle cleansing, moisturizing, and avoiding irritants. For infections, topical or oral antibiotics may be needed. Persistent or severe skin problems should be evaluated by a dermatologist."
        },
        
        # Mental Health
        'mental_health': {
            'keywords': ['anxiety', 'depression', 'stress', 'mental health', 'mood', 'panic', 'sleep problems', 'insomnia'],
            'answer': "Mental health is as important as physical health. Anxiety and depression are common and treatable conditions. Management may include therapy, medication, lifestyle changes, stress reduction techniques, and social support. Good sleep hygiene, regular exercise, and healthy relationships are important. Don't hesitate to seek professional help from a mental health provider."
        },
        
        # Cardiovascular
        'cardiovascular': {
            'keywords': ['heart', 'blood pressure', 'hypertension', 'chest pain', 'heart attack', 'stroke', 'cholesterol'],
            'answer': "Cardiovascular health involves the heart and blood vessels. High blood pressure and high cholesterol are major risk factors for heart disease and stroke. Management includes a healthy diet, regular exercise, not smoking, limiting alcohol, and managing stress. Medications may be needed. Seek immediate medical attention for chest pain, severe headache, or stroke symptoms."
        },
        
        # Diabetes and Metabolism
        'diabetes_metabolism': {
            'keywords': ['diabetes', 'blood sugar', 'glucose', 'insulin', 'metabolism', 'weight loss', 'weight gain', 'thyroid'],
            'answer': "Metabolic conditions like diabetes affect how the body processes nutrients. Type 1 diabetes requires insulin therapy, while Type 2 diabetes may be managed with diet, exercise, and medications. Blood sugar monitoring is important. Thyroid disorders affect metabolism and energy levels. Weight management involves balanced nutrition and regular physical activity."
        },
        
        # Women's Health
        'womens_health': {
            'keywords': ['menstruation', 'period', 'pregnancy', 'menopause', 'pms', 'reproductive health', 'breast health'],
            'answer': "Women's health encompasses reproductive health throughout life stages. Regular gynecological care is important. Menstrual irregularities, severe PMS, or menopausal symptoms can often be managed with lifestyle changes or medical treatments. Pregnancy requires prenatal care. Breast self-exams and regular mammograms are important for breast health."
        },
        
        # Children's Health
        'pediatric': {
            'keywords': ['children', 'baby', 'infant', 'toddler', 'pediatric', 'vaccination', 'growth', 'development'],
            'answer': "Children's health involves growth, development, and prevention. Regular pediatric checkups, vaccinations, and developmental screenings are essential. Common childhood issues include respiratory infections, fever, and minor injuries. Children's medication doses differ from adults. Always consult a pediatrician for children's health concerns."
        },
        
        # Emergency Symptoms
        'emergency': {
            'keywords': ['emergency', 'severe pain', 'bleeding', 'unconscious', 'difficulty breathing', 'chest pain', 'stroke symptoms'],
            'answer': "Certain symptoms require immediate medical attention: severe chest pain, difficulty breathing, signs of stroke (sudden weakness, speech problems, severe headache), severe bleeding, loss of consciousness, severe allergic reactions, or severe abdominal pain. When in doubt, call emergency services or go to the nearest emergency room."
        }
    }
    
    # Find matching medical topic
    for topic, info in medical_knowledge.items():
        if any(keyword in question_lower for keyword in info['keywords']):
            return info['answer']
    
    return None

def generate_ai_medical_response(question):
    """Generate AI response with improved prompting for medical questions"""
    if model is None or tokenizer is None:
        return None
    
    try:
        # Create a more comprehensive medical prompt
        medical_prompt = f"""Medical Question: {question}

Provide a helpful, accurate response about this health topic. Include:
- Brief explanation of the condition/topic
- Common symptoms or causes if relevant  
- General treatment approaches or management strategies
- When to seek medical care

Response:"""
        
        # Tokenize with appropriate length
        inputs = tokenizer.encode(medical_prompt, return_tensors="pt", max_length=300, truncation=True)
        inputs = inputs.to(model.device)
        
        # Generate with parameters optimized for informative responses
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,  # Longer responses for more detail
                min_length=inputs.shape[1] + 20,
                do_sample=True,
                top_k=30,
                top_p=0.8,
                temperature=0.6,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.1,
                early_stopping=True
            )
        
        # Decode and clean response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "Response:" in full_response:
            response = full_response.split("Response:")[-1].strip()
        else:
            response = full_response[len(medical_prompt):].strip()
        
        # Clean and validate response
        response = clean_medical_response(response)
        
        if len(response.strip()) > 15 and not is_poor_quality_response(response):
            return response
        
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
    
    return None

def clean_medical_response(text):
    """Enhanced cleaning for medical responses"""
    # Remove common artifacts and improve formatting
    text = re.sub(r'\(ABSTRACT TRUNCATED.*?\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'ABSTRACT TRUNCATED.*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Medical Question:.*?Response:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\.\s*\.\s*\.', '.', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^[^A-Za-z]*', '', text)  # Remove leading non-letters
    
    # Take meaningful sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    good_sentences = []
    
    for sentence in sentences[:4]:  # Up to 4 sentences
        if (len(sentence) > 15 and 
            not re.search(r'truncated|abstract|reference|disclaimer', sentence, re.IGNORECASE) and
            any(word in sentence.lower() for word in ['health', 'medical', 'treatment', 'symptoms', 'condition', 'doctor', 'care', 'therapy', 'disease', 'infection', 'pain', 'medication'])):
            good_sentences.append(sentence)
    
    if good_sentences:
        return '. '.join(good_sentences) + '.'
    
    return text.strip()

def is_poor_quality_response(text):
    """Enhanced quality checking"""
    if len(text.strip()) < 20:
        return True
    
    poor_indicators = [
        'abstract truncated',
        'truncated at',
        'reference',
        'citation',
        text.count('?') > 3,
        not any(char.isalpha() for char in text),
        text.lower().count('the') > len(text.split()) * 0.15,  # Too many "the"s
    ]
    
    return any(indicator if isinstance(indicator, bool) else 
              (indicator.lower() in text.lower() if isinstance(indicator, str) else indicator)
              for indicator in poor_indicators)

def get_contextual_fallback(question):
    """Provide more specific fallback responses based on question context"""
    question_lower = question.lower()
    
    # Analyze question for medical context
    if any(word in question_lower for word in ['medicine', 'drug', 'medication', 'prescription', 'dosage']):
        return ("Medications should always be taken as prescribed by a healthcare provider. Dosages depend on individual factors like age, weight, and medical condition. Never adjust medication doses without consulting your doctor or pharmacist. Be aware of potential side effects and drug interactions.")
    
    elif any(word in question_lower for word in ['diet', 'nutrition', 'food', 'eating', 'weight']):
        return ("Nutrition plays a crucial role in health. A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Specific dietary needs vary by individual, age, and health conditions. For personalized nutrition advice, consider consulting a registered dietitian.")
    
    elif any(word in question_lower for word in ['exercise', 'fitness', 'workout', 'physical activity']):
        return ("Regular physical activity is important for overall health. Most adults should aim for at least 150 minutes of moderate-intensity aerobic activity per week, plus muscle-strengthening activities. Start slowly and gradually increase intensity. Consult a healthcare provider before beginning a new exercise program, especially if you have health conditions.")
    
    elif any(word in question_lower for word in ['prevention', 'prevent', 'avoid', 'risk']):
        return ("Disease prevention often involves lifestyle modifications such as maintaining a healthy diet, regular exercise, not smoking, limiting alcohol consumption, managing stress, getting adequate sleep, and staying up-to-date with vaccinations and health screenings. Prevention strategies may vary based on individual risk factors.")
    
    else:
        return ("This appears to be a health-related question that requires personalized medical advice. While general health information can be helpful, individual circumstances vary greatly. For accurate diagnosis, treatment recommendations, and medical advice tailored to your specific situation, please consult with a qualified healthcare professional.")

def get_medical_response(user_question):
    """Enhanced medical response system with multiple fallback layers"""
    try:
        # Layer 1: Direct knowledge base match
        direct_answer = get_comprehensive_medical_answer(user_question)
        if direct_answer:
            return direct_answer + "\n\nDisclaimer: This is general health information. Please consult healthcare professionals for personalized medical advice."
        
        # Layer 2: AI-generated response
        ai_response = generate_ai_medical_response(user_question)
        if ai_response:
            return ai_response + "\n\nDisclaimer: This is AI-generated health information. Please consult healthcare professionals for personalized medical advice."
        
        # Layer 3: Contextual fallback
        contextual_response = get_contextual_fallback(user_question)
        return contextual_response + "\n\nDisclaimer: This is general information. Please consult healthcare professionals for personalized medical advice."
        
    except Exception as e:
        logger.error(f"Error generating medical response: {e}")
        return ("I understand you have a health-related question. For the most accurate and personalized medical information, please consult with a qualified healthcare professional who can properly evaluate your specific situation.\n\n"
                "Disclaimer: Please consult healthcare professionals for medical advice.")

# Enhanced test function
def test_model():
    test_questions = [
        "Do antibiotics help in the treatment of bacterial meningitis?",
        "How to treat fever?",
        "What are the symptoms of COVID-19?",
        "How to treat a headache?",
        "What causes high blood pressure?",
        "How to manage diabetes?",
        "What are the signs of depression?",
        "How to lose weight safely?",
        "What should I eat for better heart health?",
        "When should I see a doctor for chest pain?",
        "How to prevent kidney stones?",
        "What are the side effects of ibuprofen?",
        "How much exercise do I need per week?",
        "What are the symptoms of a stroke?",
        "How to improve my sleep quality?"
    ]
    
    print("Testing enhanced medical AI responses...")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = get_medical_response(question)
        print(f"A: {response}")
        print("-" * 80)

if __name__ == "__main__":
    test_model()