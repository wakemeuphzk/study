import tensorflow as tf
from poetry.model import rnn_model
from poetry.poems import process_poems
import numpy as np

start_token = 'B'
end_token = 'E'
model_dir = './modelpath/'
corpus_file = './data/poems.txt'

lr = 0.0002


# predict就相当于output（由一系列的h拼凑成）
def to_word(predict, vocabs):
    # predict[0]的形状为[0.3,0.9,0.6]，按理说长度小于vocabularies的长度，
    predict = predict[0]
    # 归一化处理
    predict /= np.sum(predict)
    # sample的返回值为标签，0到len(predict)之间的整数
    sample = np.random.choice(np.arange(len(predict)), p=predict)
    if sample > len(vocabs):
        return vocabs[-1]
    else:
        return vocabs[sample]


def gen_poem(begin_word):
    batch_size = 1
    print('## loading corpus from %s' % model_dir)
    poems_vector, word_int_map, vocabularies = process_poems(corpus_file)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=64, learning_rate=lr)

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)

        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)

        x = np.array([list(map(word_int_map.get, start_token))])
        # predict相当于output,形状是[batch_size,step,cell_num]
        # batch_size = 1,上述形状就变为[step,cell_num]
        # predict的形状就如下所示：
        # [[0.3,0.9,0.6],
        #  [0.8,0.5,0.7]]
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        if begin_word:
            word = begin_word
        else:
            word = to_word(predict, vocabularies)
        poem_ = ''

        i = 0
        while word != end_token:
            poem_ += word
            i += 1
            # 生成的诗的字数不超过24
            if i >= 24:
                break
            # x为[[ 0]],x[0, 0]=0
            x = np.zeros((1, 1))
            # word_int_map[word]的位置index赋值给x[0, 0]，比如‘雨’对应的index为100，那么x[0, 0]初始值就是100
            x[0, 0] = word_int_map[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, vocabularies)

        return poem_


def pretty_print_poem(poem_):
    poem_sentences = poem_.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


if __name__ == '__main__':
    begin_char = input('## please input the first character:')
    poem = gen_poem(begin_char)
    pretty_print_poem(poem_=poem)