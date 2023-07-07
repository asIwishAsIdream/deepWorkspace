import os
import operator
import pickle
import numpy as np

# 데이터를 읽어오는 함수
def read_data(filename):
    sents = []
    sent = []
    # 이 코드는 파일의 각 줄에 대해 반복한다
    for line in open(filename):
        # 문자열의 앞뒤에 있는 공백을 제거
        line = line.strip()
        
        # line에 어떤 값이 들어 있으면 실행하지 않는다
        # line에 아무것도 없어야 실행할 수 있다
        # 그래서 빈줄을 Test.txt에도 추가할 수 있게 한다
        if not line:
            sents.append(sent)
            sent = []
            continue
        
        # 이미 전처리된 train.txt를 가져오는 것.
        # 그래서 순서대로 단어, 태그(품사 태그), 청크(), 라벨 넣어준다
        # 청크 Chunk : 문장의 일부분 ex the black cat = 명사구 청크(Noun Phrase)가 될 수 있다
        word, tag, chunk, label = line.split(' ')
        word = word.lower()

        # 튜플 형태로 sent에 넣어준다
        sent.append((word, tag, chunk, label))
        # 빈줄을 만나면 한번에 넣어준다
        
    return sents

# main을 보면 알겠지만 train에는 위에 있는 sents가 담긴다
# 말 그대로 단어장을 만들어 주는 function이다
def make_vocab(train):
    word_dict = {}
    tag_dict = {}
    label_dict = {}
    # sent는 (word, tag, chunk, label)이 여러개 있는 list이다
    for sent in train:
        # word, tag, chunk, label
        words = [item[0] for item in sent]
        tags = [item[1] for item in sent]
        labels = [item[3] for item in sent]

        # words tags labels가 w t l 에 각각 담겨진다
        for w, t, l in zip(words, tags, labels):
            # 그래서 각 w t l 이 있으면 dictionrary에 키 : 값 으로 추가하는데 값에 +1 을 해주고 없으면 0으로 넣어준다
            if w not in word_dict:
                word_dict[w] = 0
            word_dict[w] += 1

            if t not in tag_dict:
                tag_dict[t] = 0
            tag_dict[t] += 1

            if l not in label_dict:
                label_dict[l] = 0
            label_dict[l] += 1

    # 내림차순으로 큰 count 값이 나오도록 한다(많이 나온 단어 순으로 정렬)
    sorted_words = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_tags = sorted(tag_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_labels = sorted(label_dict.items(), key=operator.itemgetter(1), reverse=True)

    # 단어 태그 라벨을 고유한 정수로 매핑한다
    # i + 2 인 이유는 PAD와 UNK를 추가하기 위해서이다
    word2id = {w: i + 2 for i, (w, c) in enumerate(sorted_words)}
    tag2id = {w: i + 2 for i, (w, c) in enumerate(sorted_tags)}
    label2id = {w: i for i, (w, c) in enumerate(sorted_labels)}

    # Padding 값 그러니까 문장의 수를 맞춰주기 위한 Padding 값은 0으로 지정한다
    # UNK는 알려지지 않은 값 그러니까 dictionary에 없는 새로운 데이터를 의미한다
    word2id['<PAD>'] = 0
    word2id['<UNK>'] = 1

    tag2id['<PAD>'] = 0
    tag2id['<UNK>'] = 1

    return word2id, tag2id, label2id

# sent는 읽어온 데이터(word, tag, chunk, label)
# 0 1 2 매개변수는 vocab에서 가져온 word tage label인데 정수로 매핑되어 있는 dic이다
def convert_ids(word2id, tag2id, label2id, sent, UNK=1):
    # word, tag, chunk, label
    words = [item[0] for item in sent]
    tags = [item[1] for item in sent]
    labels = [item[3] for item in sent]

    # 읽어온 sent의 단어 태그 라벨이 vocab 안에 있으면 해당 인덱스 번호를 가지고 단어장에 없으면 UNK 값을 넣는다
    word_ids = [word2id[w] if w in word2id else UNK for w in words]
    tag_ids = [tag2id[t] if t in tag2id else UNK for t in tags]
    label_ids = [label2id[l] for l in labels]

    return word_ids, tag_ids, label_ids, words

# 단어 embedding : 단어를 컴퓨터가 이해하고 처리할 수 있는 수치화된 벡터 형태로 변환하는 것
# lookup table : 특정 작업을 빠르게 수행하기 위해 미리 계산된 결과나 정보를 저장하는 데이터 구조
# 일종의 사전 => 특정 키 값에 대응하는 값을 빠르게 찾는 데 사용된다
# 주로 단어 임베딩을 저장함 각 단어(토큰)는 키가 되고, 해당 단어의 벡터 표현이 값이 된다
# 그래서 단어=> 벡터 표현 값으로 빠르게 찾아 갈 수 있게 된다.
# 토큰 : 토큰화는 텍스트를 의미 있는 단위로 분리하는 과정 "I love ice cream."을 단어 단위로 토큰화하면 "I", "love", "ice", "cream"이라는 4개의 토큰이 생성
def build_pretrained_lookup_table(word2id, pretrained_emb_file, embedding_dim=100):
    embedding_dict = {}
    # 이미 임베딩된 단어 in vocab : vector 값 을 dictionary에 넣는다
    for line in open(pretrained_emb_file, encoding='utf-8'):
        tokens = line.split(' ')
        # a = ['a', 'b', 'c', 'd', 'e']  =>  a[ 1 :  ]  => ['b', 'c', 'd', 'e']
        word, vecs = tokens[0], np.asarray(tokens[1:], np.float32)
        embedding_dict[word] = vecs

    # 임베딩 테이블을 초기화할 때 사용되는 범위를 설정
    scale = np.sqrt(3.0 / embedding_dim)

    # 단어 사전의 크기, 즉 고유한 단어의 수
    vocab_size = len(word2id)

    # 임베딩 행렬을 초기화한다
    # -scale ~ scale 균일 분포를 따르는 vocab_size x embedding_dim 크기의 행렬을 생성한다
    # 여기서는 임베딩으로 단어에 대한 벡터 값을 -1 ~ 1 값으로 랜덤 초기화해준다
    # 이렇게 하면 vocab에 있는 우리가 가진 모든 단어 * embedding_dim(100) 차원의 행렬이 만들어진다
    embedding = np.random.uniform(-scale, scale, [vocab_size, embedding_dim]).astype(np.float32)

    # 우리의 단어가 "glove.6B.100d.txt"에 존재하면 해당 벡터 값이 embedding에 들어가게된다
    for word, index in word2id.items():
        if word in embedding_dict:
            vec = embedding_dict[word]
            embedding[index, :] = vec

    return embedding
# build_pretrained_lookup_table => 이미 존재하는 embedding 표를 가지고 vacab에 있는 word의 벡터 값을
# 가져와서 embedding_dict에 단어 : 벡터값으로 저장한다 
# 그리고 사용한 embedding 을 만들어 준다. 우리 단어가 embedding_dict에 존재하면 해당 벡터 값을 embedding에 담고 없으면 그냥 랜덤한 값으로 초기화해서 담아준다


if __name__ == "__main__":
    # print(os.getcwd())
    traindata = read_data(os.path.join("data", "train.txt"))
    devdata = read_data(os.path.join("data", "valid.txt"))
    testdata = read_data(os.path.join("data", "test.txt"))
    
    word2id, tag2id, label2id = make_vocab(traindata)

    train = [convert_ids(word2id, tag2id, label2id, sent) for sent in traindata]
    dev = [convert_ids(word2id, tag2id, label2id, sent) for sent in devdata]
    test = [convert_ids(word2id, tag2id, label2id, sent) for sent in testdata]

    embedding = build_pretrained_lookup_table(word2id, os.path.join("data", "glove.6B.100d.txt"))

    data = {'train': train, 'dev': dev, 'test': test, 'w2id': word2id, 't2id': tag2id, 'l2id': label2id, 'embedding': embedding}

    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
