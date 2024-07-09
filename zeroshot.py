from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")
sequence_to_classify = "La victoria de la selección de fútbol de argentina fue por la tanda de penales ante la selección de Ecuador"
candidate_labels = ['ciencia', 'deportes', 'politicas']
resultado_clasificacion = classifier(sequence_to_classify, candidate_labels)
print(resultado_clasificacion)

#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}
