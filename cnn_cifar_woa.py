import os
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,datasets
import numpy as np



conv_layers = [
    # unit 1
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(64,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 2
    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(128,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 3
    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(256,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 4
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same'),

    # unit 5
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.Conv2D(512,kernel_size=[3,3],padding='same',activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=[2,2],strides=2,padding='same')

]


# 主要超参数
learning_rate = 1e-4
epochs = 100
batch_size = 128
cutmix_prob = 0.5
early_stopping_patience = 10

log_dir = './cnn_logs_woa/'
summary_writer_train = tf.summary.create_file_writer(os.path.join(log_dir, 'train'))
summary_writer_test = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))




def preprocess(x,y):  # 预处理函数
    x = tf.cast(x,dtype=tf.float32) / 255
    y = tf.cast(y,dtype=tf.int64)
    return x,y



def cutmix(x, y, alpha=1.0):
    batch_size = tf.shape(x)[0]
    indices = tf.random.shuffle(tf.range(batch_size))
    shuffled_x = tf.gather(x, indices)
    shuffled_y = tf.gather(y, indices)
    
    lambda_ = np.random.beta(alpha, alpha)  # beta分布
    cut_rat = tf.sqrt(1.0 - lambda_)
    cut_w = tf.cast(32 * cut_rat, tf.int32)
    cut_h = tf.cast(32 * cut_rat, tf.int32)
    
    # 随机选择切割区域的[中心点]
    cx = tf.random.uniform([], 0, 32, tf.int32)
    cy = tf.random.uniform([], 0, 32, tf.int32)

    bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, 32)
    bby1 = tf.clip_by_value(cy - cut_h // 2, 0, 32)
    bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, 32)
    bby2 = tf.clip_by_value(cy + cut_h // 2, 0, 32)

    new_x, shuffled_x = x.numpy(), shuffled_x.numpy()
    new_x[:, bbx1:bbx2, bby1:bby2, :] = shuffled_x[:, bbx1:bbx2, bby1:bby2, :]

    lambda_ = 1.0 - (bbx2 - bbx1) * (bby2 - bby1) / (32 * 32)
    
    y_onehot = tf.one_hot(y, 100).numpy()
    shuffled_y_onehot = tf.one_hot(shuffled_y, 100).numpy()
    new_y = lambda_ * y_onehot + (1.0 - lambda_) * shuffled_y_onehot

    return tf.convert_to_tensor(new_x), tf.convert_to_tensor(new_y)



def load_data():
    (x,y),(x_test,y_test) = datasets.cifar100.load_data()
    y = tf.reshape(y,[50000])
    y_test = tf.reshape(y_test,[10000])
    train_db = tf.data.Dataset.from_tensor_slices((x,y))
    train_db = train_db.shuffle(10000).map(preprocess).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    test_db = test_db.map(preprocess).batch(batch_size)
    return train_db, test_db



def train(train_db, test_db):
    
    best_test_accuracy = 0.0
    epochs_without_improvement = 0
    
    conv_net = Sequential(conv_layers)   # [b,32,32,3] => [b,1,1,512]
    
    fc_net = Sequential([
        layers.Dense(256,activation=tf.nn.relu),
        layers.Dense(128,activation=tf.nn.relu),
        layers.Dense(100,activation=None)
    ])
    conv_net.build(input_shape=[None, 32, 32, 3])
    fc_net.build(input_shape=[None,512])

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    variables = conv_net.variables + fc_net.variables

    for epoch in range(epochs):
        
        train_loss_avg = tf.metrics.Mean()
        train_accuracy = tf.metrics.Accuracy()

        for step,(x,y) in enumerate(train_db):
            # Cutmix数据增强
            if tf.random.uniform([]) < cutmix_prob: x, y = cutmix(x, y)
            else: y = tf.one_hot(y, 100)
            with tf.GradientTape() as tape:
                out = conv_net(x)
                out = tf.reshape(out,[-1,512])  # flatten
                logits = fc_net(out)
                
                # compute loss
                loss = tf.losses.categorical_crossentropy(y,logits,from_logits=True)
                loss = tf.reduce_mean(loss)

            grads = tape.gradient(loss,variables)
            optimizer.apply_gradients(zip(grads,variables))
            
            train_loss_avg.update_state(loss)
            train_accuracy.update_state(tf.argmax(y, axis=1), tf.argmax(logits, axis=1))
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {train_loss_avg.result()}")
        
        with summary_writer_train.as_default():
            tf.summary.scalar('loss', train_loss_avg.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        
        test_accuracy = evaluate_model(conv_net, fc_net, test_db, epoch)
        
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            epochs_without_improvement = 0
            save_path = os.path.join(log_dir, 'best_model.h5')
            conv_net.save_weights(save_path)
            print(f"Saved model at epoch {epoch} with test accuracy {best_test_accuracy}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == early_stopping_patience:
                print(f"No improvement for {early_stopping_patience} epochs, stopping training.")
                break
    
    
def evaluate_model(conv_net, fc_net, test_db, epoch):
    test_loss_avg = tf.metrics.Mean()
    test_accuracy = tf.metrics.Accuracy()

    for x,y in test_db:
        out = conv_net(x)
        out = tf.reshape(out,[-1,512])
        logits = fc_net(out)
        
        loss = tf.losses.categorical_crossentropy(tf.one_hot(y, 100), logits, from_logits=True)
        test_loss_avg.update_state(loss)
        test_accuracy.update_state(tf.argmax(logits, axis=1), y)
    
    with summary_writer_test.as_default():
        tf.summary.scalar('loss', test_loss_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
        
    print("\n**********************************************************************************************")
    print(f"Epoch {epoch}, Test Loss: {test_loss_avg.result()}, Test Accuracy: {test_accuracy.result()}")
    print("**********************************************************************************************\n\n")
    return test_accuracy.result()


if __name__ == '__main__':
    train_db, test_db = load_data()
    train(train_db, test_db)
