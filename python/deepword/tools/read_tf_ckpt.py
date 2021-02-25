import fire
import tensorflow as tf


def extracting(ckpt):
    ckpt_reader = tf.train.load_checkpoint(ckpt)
    tf_vars = ckpt_reader.get_variable_to_shape_map()
    tf_var_names = sorted(tf_vars.keys())
    for v_name in tf_var_names:
        print("v_name: {}, v_shape: {}".format(v_name, tf_vars[v_name]))
        print(ckpt_reader.get_tensor(v_name))
        print("--------------")


if __name__ == "__main__":
    fire.Fire(extracting)
