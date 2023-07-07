import torch
from torch.nn.utils.rnn import pad_sequence
import os

# 청크 Chunk : 문장의 일부분 ex the black cat = 명사구 청크(Noun Phrase)가 될 수 있다
def get_chunk_type(tag_name):
    tag_class = tag_name.split('-')[0]
    tag_type = tag_name.split('-')[-1]
    return tag_class, tag_type


def get_chunks(seq):
    # seq에 답긴 값은 "O" B-ORG B-MISC 이런 lable(정답) 값이다
    default = "O"

    chunks = []
    chunk_type, chunk_start = None, None
    # 연속되는 chunk를 가질 경우 chunk_start : chunk_end로 묶어 줘야한다
    # 이는 예제를 가지고 코드의 흐름을 봐야한다 Goorm 메모장 10p에 이 과정을 넣었으니 잊었을 때 볼 수 있도록.
    for i, tok in enumerate(seq):
        # End of a chunk 1
        # batch_size 만큼 반복적으로 get_chunks가 호출되기 때문에 seq가 "O" 일 경우 chunk_type은 계속 None이 된다.
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            # B                 ORG 가 담긴다
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok)
            
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass

    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

# ii는 인덱스 번호 => batch_size만큼 반복하는 횟수
# num_data는 len(data) 인데 여기에 들어오는 data는 train, test, validation의 데이터이다
# batch size만큼 data의 크기를 나눠준다. 이때 ii가 그 1 epoch를 돌리기 위한 수가 되겠죠. 즉 ii의 총 길이 = data // batch_size
def batchify(ii, batch_size, num_data, data):
    # 배치 사이즈 만큼 데이터를 나눠주는 코드
    start = ii * batch_size
    end = num_data if (ii + 1) * batch_size > num_data else (ii + 1) * batch_size

    batch_data = data[start:end]

    batch_word_ids = [torch.tensor(data[0], dtype=torch.long) for data in batch_data]
    batch_tag_ids = [torch.tensor(data[1], dtype=torch.long) for data in batch_data]
    batch_labels_ids = [torch.tensor(data[2], dtype=torch.long) for data in batch_data]

    batch_word_ids = pad_sequence(batch_word_ids, batch_first=True)
    batch_tag_ids = pad_sequence(batch_tag_ids, batch_first=True)
    batch_labels_ids = pad_sequence(batch_labels_ids, batch_first=True)

    return batch_word_ids, batch_tag_ids, batch_labels_ids

# ner : Named Entity Recognition - 텍스트에서 특정 카테고리에 속하는 단어나 구를 식별하고 분류하는 과정
# F1 점수는 분류 문제에서 모델의 성능을 평가하는 방법 중 하나입니다. 정밀도(Precision)와 재현율(Recall)의 조화 평균을 의미
# F1 점수는 모델이 높은 정밀도와 재현율을 동시에 달성하도록 만드는 데 도움이 됨
def evaluate_ner_F1(total_answer_ids, total_pred_ids, id2label):
    num_match = num_preds = num_answers = 0
    # total_answer_ids는 배치 사이즈에 있는 데이터의 정답값
    # 그러니까 예를 들어 train.txt에서 label을 index로 바꾸고 배열을 변경해서 index:label로 저장한 id2label
    # 그리고 배치사이즈에 해당되는 정답 index와 예측 index가 저장된 total
    # 이것들의 index를 id2label에 넣어서 둘이 비교 및 이해하기 쉽게 해준다
    for answer_ids, pred_ids in zip(total_answer_ids, total_pred_ids):
        # 실제 정답 lable과 예측 lable이 담기게 된다
        answers = [id2label[l_id] for l_id in answer_ids]
        preds = [id2label[l_id] for l_id in pred_ids]

        # 그래서 예측값과 정답 lable에 대한 chunks를 얻어와라
        answer_seg_result = set(get_chunks(answers))
        pred_seg_result = set(get_chunks(preds))

        num_match += len(answer_seg_result & pred_seg_result)
        num_answers += len(answer_seg_result)
        num_preds += len(pred_seg_result)

    precision = 100.0 * num_match / num_preds
    recall = 100.0 * num_match / num_answers
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1


def evaluate_ner_F1_and_write_result(total_words, total_answer_ids, total_pred_ids, id2label, setname):
    num_match = num_preds = num_answers = 0

    filename = os.path.join("save", "best_%s_result.txt" % setname)
    of = open(filename, "w")
    for words, answer_ids, pred_ids in zip(total_words, total_answer_ids, total_pred_ids):
        answers = [id2label[l_id] for l_id in answer_ids]
        preds = [id2label[l_id] for l_id in pred_ids]

        answer_seg_result = set(get_chunks(answers))
        pred_seg_result = set(get_chunks(preds))

        num_match += len(answer_seg_result & pred_seg_result)
        num_answers += len(answer_seg_result)
        num_preds += len(pred_seg_result)

        for w, a_l, p_l in zip(words, answers, preds):
            of.write("\t".join([w, a_l, p_l]) + "\n")
        of.write("\n")
    of.close()

    precision = 100.0 * num_match / num_preds
    recall = 100.0 * num_match / num_answers
    F1 = 2 * precision * recall / (precision + recall)

    return precision, recall, F1
