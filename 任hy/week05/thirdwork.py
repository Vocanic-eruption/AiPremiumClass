from collections import defaultdict
import matplotlib.pyplot as plt

def parse_cooking_data(file_path):
    label_counts = defaultdict(int)
    questions_by_label = defaultdict(list)
    total_questions = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split(' ', 1)
            labels = [part for part in parts[0].split(' ') if part.startswith('__label__')]
            question = parts[1] if len(parts) > 1 else ""
            
            for label in labels:
                label_name = label.replace('__label__', '')
                label_counts[label_name] += 1
                questions_by_label[label_name].append(question)
            
            total_questions += 1
    
    return label_counts, questions_by_label, total_questions

def analyze_data(label_counts, questions_by_label, total_questions):
    print(f"Total questions: {total_questions}")
    print(f"Unique labels: {len(label_counts)}")
    
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 most common labels:")
    for label, count in sorted_labels[:20]:
        print(f"{label}: {count} ({count/total_questions:.1%})")
    
    labels_per_question = []
    with open('cooking.stackexchange.txt', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            labels = [part for part in parts[0].split(' ') if part.startswith('__label__')]
            labels_per_question.append(len(labels))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    top_labels = [x[0] for x in sorted_labels[:10]]
    top_counts = [x[1] for x in sorted_labels[:10]]
    plt.barh(top_labels[::-1], top_counts[::-1])
    plt.title('Top 10 Most Common Labels')
    plt.xlabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(labels_per_question, bins=range(1, max(labels_per_question)+2), align='left', rwidth=0.8)
    plt.title('Labels per Question Distribution')
    plt.xlabel('Number of Labels')
    plt.ylabel('Count')
    plt.xticks(range(1, max(labels_per_question)+1))
    
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'cooking.stackexchange.txt'
    label_counts, questions_by_label, total_questions = parse_cooking_data(file_path)
    analyze_data(label_counts, questions_by_label, total_questions)

if __name__ == "__main__":
    main()
