import json
import random


def divide_data():
    # 1. Read all original response logs
    with open('data/ednet/log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    # 2. Flatten all response records into individual logs (not grouped by student)
    all_logs = []
    for stu in stus:
        user_id = stu['user_id']
        for log in stu['logs']:
            all_logs.append({
                'user_id': user_id,
                'exer_id': log['exer_id'],
                'score': log['score'],
                'knowledge_code': log['knowledge_code']
            })

    # 3. Global shuffle (completely random)
    random.shuffle(all_logs)
    total = len(all_logs)

    # 8:2 split
    train_size = int(total * 0.8)
    test_size = total - train_size

    train_set = all_logs[:train_size]
    test_set = all_logs[train_size:]

    # 4. Save
    with open('data/ednet/train_set.json', 'w', encoding='utf8') as f:
        json.dump(train_set, f, indent=4, ensure_ascii=False)
    with open('data/ednet/test_set.json', 'w', encoding='utf8') as f:
        json.dump(test_set, f, indent=4, ensure_ascii=False)

    print(f'Total response records: {total}')
    print(f'train: {len(train_set)} (80%), test: {len(test_set)} (20%)')
    print('Split completed!')


if __name__ == '__main__':
    divide_data()