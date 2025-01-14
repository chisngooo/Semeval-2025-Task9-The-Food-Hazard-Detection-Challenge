import pandas as pd
import json
import re
file_path = '../data/data.csv'
output_file = "ss_test_512.json"
df = pd.read_csv(file_path)
data = df.to_dict(orient='records')

def clean_text(text):
    text = re.sub(r'[^\w\s.,!?;]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.,!?;])', r'\1 ', text)
    return text.strip()


def chunk_text_by_sentence(text, chunk_size=400):
    text = clean_text(text)
    text = re.sub(r'\n', ' ', text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) <= chunk_size:
            current_chunk.append(sentence)
            current_word_count += len(words)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_word_count = len(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

result = []
for idx, item in enumerate(data):
    text = item.get("text", "")
    chunks = chunk_text_by_sentence(text)
    hazard_category = item.get("hazard-category", "")
    product_category = item.get("product-category", "")
    for chunk_id, chunk in enumerate(chunks):
        full_text = f"Year: {item.get('year')}, Month: {item.get('month')}, Day: {item.get('day')}, Country: {item.get('country')}, Title: {item.get('title')} - {chunk}"
        result.append({
            "stt": idx,
            "chunk_id": f"{idx}_{chunk_id}",
            "text": full_text,
            "hazard_category": hazard_category,
            "product_category": product_category
        })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print(f"Dữ liệu đã được chunk và lưu vào '{output_file}'.")