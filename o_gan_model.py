import tensorflow as tf
import ops

class O_GAN:
	'''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''
	def __init__(self, options):
		self.options = options

		self.g_bn0 = ops.batch_norm(name='g_bn0')
		self.g_bn1 = ops.batch_norm(name='g_bn1')
		self.g_bn2 = ops.batch_norm(name='g_bn2')
		self.g_bn3 = ops.batch_norm(name='g_bn3')

		self.d_bn1 = ops.batch_norm(name='d_bn1')
		self.d_bn2 = ops.batch_norm(name='d_bn2')
		self.d_bn3 = ops.batch_norm(name='d_bn3')
		self.d_bn4 = ops.batch_norm(name='d_bn4')


	def build_model(self):
		img_size = self.options['image_size']
		t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
		t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

		fake_image = self.generator(t_z, t_real_caption)
		
		disc_real_image, disc_real_image_logits   = self.discriminator(t_real_image, t_real_caption)
		disc_wrong_image, disc_wrong_image_logits   = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
		disc_fake_image, disc_fake_image_logits   = self.discriminator(fake_image, t_real_caption, reuse = True)

		mean_disc_real_image = tf.reduce_mean(disc_real_image, 1, keep_dims = True)
		mean_wrong_real_image = tf.reduce_mean(disc_wrong_image, 1, keep_dims = True)
		mean_disc_fake_image = tf.reduce_mean(disc_fake_image, 1, keep_dims = True)

		print(mean_disc_real_image.get_shape())
		print(mean_disc_fake_image.get_shape())

		z1_corr = ops.correlation(t_real_image, fake_image)
		z2_corr = ops.correlation(t_real_image, t_wrong_image)

		# d_loss = 2 * mean_disc_real_image - mean_wrong_real_image - mean_disc_fake_image - self.options['lam'] * (z1_corr + z2_corr)
		# d_loss = tf.reduce_mean(d_loss)
		# g_loss = mean_wrong_real_image + mean_disc_fake_image - self.options['lam'] * (z1_corr + z2_corr)
		# g_loss = tf.reduce_mean(g_loss)

		d_loss = 2 * mean_disc_real_image - mean_wrong_real_image - mean_disc_fake_image - 0.25 * (z1_corr + z2_corr)
		d_loss = tf.reduce_mean(d_loss)
		g_loss = mean_wrong_real_image + mean_disc_fake_image - 0.25 * (z1_corr + z2_corr)
		g_loss = tf.reduce_mean(g_loss)

		# g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_image_logits, labels = tf.ones_like(disc_fake_image)))
		#
		# d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real_image_logits, labels = tf.ones_like(disc_real_image)))
		# d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_wrong_image_logits, labels = tf.zeros_like(disc_wrong_image)))
		# d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_image_logits, labels = tf.zeros_like(disc_fake_image)))
		#
		# d_loss = d_loss1 + d_loss2 + d_loss3

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'd_' in var.name]
		g_vars = [var for var in t_vars if 'g_' in var.name]

		input_tensors = {
			't_real_image' : t_real_image,
			't_wrong_image' : t_wrong_image,
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}

		variables = {
			'd_vars' : d_vars,
			'g_vars' : g_vars
		}

		loss = {
			'g_loss' : g_loss,
			'd_loss' : d_loss
		}

		outputs = {
			'generator' : fake_image
		}

		checks = {
			# 'd_loss1': d_loss1,
			# 'd_loss2': d_loss2,
			# 'd_loss3' : d_loss3,
			'mean_disc_real_image' : mean_disc_real_image,
			'mean_wrong_real_image' : mean_wrong_real_image,
			'mean_disc_fake_image' : mean_disc_fake_image,
			'z1_corr' : z1_corr,
			'z2_corr' : z2_corr
			# 'disc_wrong_image_logits' : disc_wrong_image,
			# 'disc_fake_image_logits' : disc_fake_image_logits
		}
		
		return input_tensors, variables, loss, outputs, checks

	def build_generator(self):
		img_size = self.options['image_size']
		t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
		t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
		fake_image = self.sampler(t_z, t_real_caption)
		
		input_tensors = {
			't_real_caption' : t_real_caption,
			't_z' : t_z
		}
		
		outputs = {
			'generator' : fake_image
		}

		return input_tensors, outputs

	# Sample Images for a text embedding
	def sampler(self, t_z, t_text_embedding):
		tf.get_variable_scope().reuse_variables()
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat(1, [t_z, reduced_text_embedding])
		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')
		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0, train = False))
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1, train = False))
		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2, train = False))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3, train = False))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	# GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def generator(self, t_z, t_text_embedding):
		
		s = self.options['image_size']
		s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
		
		reduced_text_embedding = ops.lrelu( ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding') )
		z_concat = tf.concat([t_z, reduced_text_embedding],1)

		print("z_concat.get_shape()",z_concat.get_shape())

		z_ = ops.linear(z_concat, self.options['gf_dim']*8*s16*s16, 'g_h0_lin')

		print("z_.get_shape()", z_.get_shape())

		h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
		h0 = tf.nn.relu(self.g_bn0(h0))
		
		h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim']*4], name='g_h1')
		h1 = tf.nn.relu(self.g_bn1(h1))
		
		h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim']*2], name='g_h2')
		h2 = tf.nn.relu(self.g_bn2(h2))
		
		h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim']*1], name='g_h3')
		h3 = tf.nn.relu(self.g_bn3(h3))
		
		h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')
		
		return (tf.tanh(h4)/2. + 0.5)

	# DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
	def discriminator(self, image, t_text_embedding, reuse=False):
		if reuse:
			tf.get_variable_scope().reuse_variables()

		h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name = 'd_h0_conv')) #32
		# h1 = ops.lrelu(tf.nn.batch_normalization(ops.conv2d(h0, self.options['df_dim'] * 2, name='d_h1_conv')))  # 16
		h1 = ops.lrelu( self.d_bn1(ops.conv2d(h0, self.options['df_dim']*2, name = 'd_h1_conv'))) #16
		h2 = ops.lrelu( self.d_bn2(ops.conv2d(h1, self.options['df_dim']*4, name = 'd_h2_conv'))) #8
		h3 = ops.lrelu( self.d_bn3(ops.conv2d(h2, self.options['df_dim']*8, name = 'd_h3_conv'))) #4

		print("h3_shape", h3.get_shape())

		d1 = tf.layers.dense(inputs = tf.layers.Flatten()(h3),
							 units=self.options['z_dim'],
							 kernel_initializer=tf.random_normal_initializer(0, 0.02),
							 name = 'd_d1_dense')

		print("d1_shape", d1.get_shape())

		# ADD TEXT EMBEDDING TO THE NETWORK
		reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
		d1_concat = tf.concat([d1, reduced_text_embeddings], 1, name='d1_concat')

		# print("reduced_text_embeddings.get_shape()", reduced_text_embeddings.get_shape())
		# print("d1_concat.get_shape()", d1_concat.get_shape())

		# reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
		# reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
		# tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')
		#
		# h3_concat = tf.concat( [h3, tiled_embeddings], 3, name='h3_concat')
		# h3_new = ops.lrelu( self.d_bn4(ops.conv2d(h3_concat, self.options['df_dim']*8, 1,1,1,1, name = 'd_h3_conv_new'))) #4
		#
		# h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')
		
		return d1_concat, tf.nn.sigmoid(d1_concat)
