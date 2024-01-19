  #the second loss term
cost2 = tf.reduce_sum(tf.reduce_min(dist, axis=1))

d = pw_distance(prototypes)
diag_ones = tf.convert_to_tensor(np.eye(k_protos, dtype=float))
diag_ones = tf.dtypes.cast(diag_ones, tf.float32)
d1 = d + diag_ones * tf.reduce_max(d)
d2 = tf.reduce_min(d1, axis=1)
min_d2_dist = tf.reduce_min(d2)
# the third loss term
cost3 = tight_pos_sigmoid_offset(min_d2_dist, 1) + 1e-8