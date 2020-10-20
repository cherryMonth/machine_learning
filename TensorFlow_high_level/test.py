summary_op = tf.summary.merge_all()
logs_train_dir = "custom_logs"  # 写入汇总数据的文件夹
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
MAX_STEP = 3000
try:
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy])

        if step % 100 == 0:
            print("Step %d, train loss = %.2f, train accuracy = %.2f" % (step, tra_loss, tra_acc))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, "model.ckpt")
            saver.save(sess, checkpoint_path, global_step=step)
except tf.errors.OutOfRangeError:
    print("Done training -- epoch limit reached.")
finally:
    coord.request_stop()
coord.join(threads)
sess.close()