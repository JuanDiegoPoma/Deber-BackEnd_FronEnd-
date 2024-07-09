from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/clasificar', methods=['POST'])
def classify_text():
    data = request.json
    texto = data.get('texto', '')
    candidate_labels = ['Política', 'Deportes', 'Religión', 'Otros']
    
    resultado_clasificacion = classifier(texto, candidate_labels)
    max_score = resultado_clasificacion['scores'].index(max(resultado_clasificacion['scores']))
    label_score = resultado_clasificacion['labels'][max_score]
    
    return jsonify({'label': label_score})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5010)
