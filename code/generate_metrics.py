import json
from sklearn.metrics import confusion_matrix, classification_report
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


label_path = "/"
predict_path = "/"
negative_word = "无法生成"

label_select_set = set()
label_data = []
label_positive = []
with open(label_path) as f:
    for idx, line in enumerate(f):
        line = eval(line).strip()
        label_data.append(list(line))
        label_positive.append(int(negative_word not in line and line != ""))
        if negative_word not in line and line != "":
            label_select_set.add(idx)

final_select_set = set()
predict_data = []
predict_positive = []
with open(predict_path) as f:
    for idx, line in enumerate(f):
        line = eval(line).strip()
        predict_data.append(list(line))
        predict_positive.append(int(negative_word not in line and line != ""))
        if idx in label_select_set and negative_word not in line and line != "":
            final_select_set.add(idx)

print(confusion_matrix(label_positive, predict_positive))
print(classification_report(label_positive, predict_positive))

final_label_data = []
final_predict_data = []
for idx in final_select_set:
    final_label_data.append([label_data[idx]])
    final_predict_data.append(predict_data[idx])

print('bleu 1-gram: %f' % corpus_bleu(final_label_data, final_predict_data, weights=(1, 0, 0, 0)))
print('bleu 2-gram: %f' % corpus_bleu(final_label_data, final_predict_data, weights=(0.5, 0.5, 0, 0)))
print('bleu 3-gram: %f' % corpus_bleu(final_label_data, final_predict_data, weights=(0.33, 0.33, 0.33, 0)))
print('bleu 4-gram: %f' % corpus_bleu(final_label_data, final_predict_data, weights=(0.25, 0.25, 0.25, 0.25)))

for i in range(len(final_label_data)):
    final_label_data[i] = ' '.join(final_label_data[i][0])
    final_predict_data[i] = ' '.join(final_predict_data[i])

rouge = Rouge()
print(json.dumps(rouge.get_scores(hyps=final_predict_data, refs=final_label_data, avg=True), indent=4))
