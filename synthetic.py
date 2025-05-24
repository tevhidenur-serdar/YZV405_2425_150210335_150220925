import argparse
import pandas as pd
import random
import os

# Generating synthetic sentences based on a template and idiomatic/literal expression
def generate_sentences(expression, templates_tr, templates_it, label, language):
    templates = templates_tr if language == "tr" else templates_it
    data = []

    # Randomly selecting 2 templates to generate diversity
    for i, temp in enumerate(random.sample(templates, k=2)):
        sentence = temp.format(expression)  # Inserting the expression into the template
        tokens = sentence.split()           # Tokenizing full sentence
        expr_tokens = expression.split()    # Tokenizing just expression
        try:
            start = tokens.index(expr_tokens[0])
            indices = list(range(start, start + len(expr_tokens)))
        except ValueError:
            indices = [-1]

        data.append({
            "id": f"synth_{label}_{expression}_{i}",
            "language": language,
            "sentence": sentence,
            "tokenized_sentence": tokens,
            "expression": expression,
            "category": label,
            "indices": indices
        })
    return data


# Main Function
def main(args):
    # Loading evaluation data
    eval_df = pd.read_csv(args.eval_path)
    idiom_language_map = dict(zip(eval_df["expression"], eval_df["language"]))
    idioms = eval_df['expression'].dropna().unique()

    # Defining templates
    idiomatic_templates_tr = [
        "O da sonunda {} zorunda kaldı.",
        "{} yüzünden tartışma çıktı.",
        "Onlar hep {} zorundaymış gibi davranıyor.",
        "Bazıları {} gerektiğini düşünüyor."
    ]

    literal_templates_tr = [
        "{} kelimesini sözlükte aradı.",
        "Cümlede geçen {} kısmını vurguladı.",
        "Öğretmen {} örneğini tahtaya yazdı.",
        "Sınavda {} deyimini açıklaması istendi."
    ]

    idiomatic_templates_it = [
        "Alla fine è stato costretto a {}.",
        "A causa di {}, è nata una discussione.",
        "Si comportano sempre come se fossero {}.",
        "Alcuni pensano che {} sia necessario."
    ]

    literal_templates_it = [
        "Ha cercato la parola {} nel dizionario.",
        "Ha evidenziato la parte {} nella frase.",
        "L'insegnante ha scritto l'esempio {} alla lavagna.",
        "Nel test è stato chiesto di spiegare {}."
    ]

    # Generating synthetic data
    synthetic_data = []
    for expr in idioms:
        lang = idiom_language_map.get(expr, "tr")  
        synthetic_data += generate_sentences(expr, idiomatic_templates_tr, idiomatic_templates_it, "idiomatic", lang)
        synthetic_data += generate_sentences(expr, literal_templates_tr, literal_templates_it, "literal", lang)

    # Converting the list of dictionaries to a DataFrame and saving as CSV
    synthetic_df = pd.DataFrame(synthetic_data)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    synthetic_df.to_csv(args.output_path, index=False)
    


# Argument Parser
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True, help="Path to eval.csv")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save synthetic_data.csv")
    args = parser.parse_args()
    main(args)
