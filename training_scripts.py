# set = h.Setup(model_name='m_abram_opt',
#                   nb_epochs=250, save_config=True, learning_rate=0.051, batch_size=5, embedding_loss=False,
#                   background_pred=True)
# set.main()
# ana.score('m_abram_opt', include_metric=True)

# """ABRAM OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10001) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()

# """ABRAM OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()


# """ABRAM OPT ROUND 1.2"""
# for i in range(1, 50):
#     cur_nb_epochs = np.random.randint(200, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='abram_1.2_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, embedding_loss=False,
#                   background_pred=True)
#     set.main()


# """AZRAEL OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='azrael_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """AZRAEL OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='azrael_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """AZRAEL OPT ROUND 2.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 250
# lr = 0.001
# bs = 3
# for emb_dim in (8, 16, 32, 64):
#     for margin in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         print('Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#         set = h.Setup(model_name='azrael2_' + str(emb_dim) + '_' + str(margin),
#                       embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs, save_config=True, learning_rate=lr,
#                       batch_size=bs, include_background=True,
#                       background_pred=False,
#                       nb_iterations=0, embedding_loss=True)
#         set.main()


# """AZRAEL OPT ROUND 3.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 250
# lr = 0.001
# bs = 3
# emb_dim = 64
# margin = 0.5
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='azrael3.1_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=lr,
#                   batch_size=bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """METRIC AZRAEL OPT ROUND 2.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 103
# lr = 0.065
# bs = 1
# for emb_dim in (8, 16, 32, 64):
#     for margin in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         print('Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#         set = h.Setup(model_name='m_azrael2_' + str(emb_dim) + '_' + str(margin),
#                       embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs, save_config=True, learning_rate=lr,
#                       batch_size=bs, include_background=True,
#                       background_pred=False,
#                       nb_iterations=0, embedding_loss=True)
#         set.main()
# ana.score('m_azrael2_', include_metric=True)

# ana.score_metric('eve_0')


# """Metric AZRAEL OPT ROUND 3.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 103
# lr = 0.065
# bs = 1
# emb_dim = 32
# margin = 0.9
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_azrael3_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=lr,
#                   batch_size=bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_azrael3_', include_metric=True)

# """Metric AZRAEL OPT ROUND 4.0"""
# bs = 1
# nb_epochs = 103
# emb_dim = 32
# margin = 0.9
# subsample_size = 36
# for lr in (0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05,
# 0.075, 0.1):
#     cur_lr = lr
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', cur_lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_azrael4_' + str(cur_lr),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=cur_lr,
#                   batch_size=bs, include_background=True,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_azrael4_', include_metric=True)

# set = h.Setup(model_name='m_azrael_opt',
#               subsample_size=36, embedding_dim=32, margin=0.9, nb_epochs=103,
#               save_config=True, learning_rate=0.005,
#               batch_size=1, include_background=True,
#               background_pred=False,
#               nb_iterations=0, embedding_loss=True)
# set.main()
# ana.score('m_azrael_opt', include_metric=True)


# """EZEKIEL OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='ezekiel_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EZEKIEL OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='ezekiel_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """EZEKIEL OPT ROUND 2.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 250
# lr = 0.001
# bs = 3
# for emb_dim in (8, 16, 32, 64):
#     for margin in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         print('Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#         set = h.Setup(model_name='ezekiel2_' + str(emb_dim) + '_' + str(margin),
#                       embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs, save_config=True, learning_rate=lr,
#                       batch_size=bs, include_background=False,
#                       background_pred=False,
#                       nb_iterations=0, embedding_loss=True)
#         set.main()


# """EZEKIEL OPT ROUND 3.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 250
# lr = 0.001
# bs = 3
# emb_dim = 64
# margin = 0.5
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='ezekiel3.1_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=lr,
#                   batch_size=bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

#"""Metric EZEKIEL OPT ROUND 2.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 180
# lr = 0.006
# bs = 1
# for emb_dim in (8, 16, 32, 64):
#     for margin in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
#         print('Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#         set = h.Setup(model_name='m_ezekiel2_' + str(emb_dim) + '_' + str(margin),
#                       embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs, save_config=True, learning_rate=lr,
#                       batch_size=bs, include_background=False,
#                       background_pred=False,
#                       nb_iterations=0, embedding_loss=True)
#         set.main()
# ana.score('m_ezekiel2_', include_metric=True)

# """Metric EZEKIEL OPT ROUND 3.0"""
# """Taking Parameters from Optimisation Round 1"""
# nb_epochs = 180
# lr = 0.006
# bs = 1
# emb_dim = 64
# margin = 0.1
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_ezekiel3_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=lr,
#                   batch_size=bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_ezekiel3_', include_metric=True)

# """Metric EZEKIEL OPT ROUND 4.0"""
# bs = 1
# nb_epochs = 180
# emb_dim = 64
# margin = 0.1
# subsample_size = 100
# for lr in (0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05,
# 0.075, 0.1):
#     cur_lr = lr
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', cur_lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_ezekiel4_' + str(cur_lr),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=cur_lr,
#                   batch_size=bs, include_background=False,
#                   background_pred=False,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_ezekiel4_', include_metric=True)

# set = h.Setup(model_name='m_ezekiel_opt',
#               subsample_size=100, embedding_dim=64, margin=0.1, nb_epochs=180,
#               save_config=True, learning_rate=0.000025,
#               batch_size=1, include_background=False,
#               background_pred=False,
#               nb_iterations=0, embedding_loss=True)
# set.main()
# ana.score('m_ezekiel_opt', include_metric=True)


# """EVE OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='eve_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EVE OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='eve_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EVE OPT ROUND 2"""
# nb_epochs = 250
# lr = 0.00045
# bs = 4
# for i in range(1, 26):
#     cur_emb_dim = np.random.randint(8, 65)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = np.random.randint(1, 10001) / 10
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='eve2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """EVE OPT ROUND 3"""
# nb_epochs = 250
# lr = 0.00045
# bs = 4
# emb_dim = 64
# margin = 0.6
# scaling = 25
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='eve3.1_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """EVE OPT ROUND 4"""
# nb_epochs = 250
# lr = 0.00045
# bs = 4
# subsample_size = 1024
# for i in range(1, 6):
#     cur_emb_dim = np.random.randint(65, 128)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = 25
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='eve2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True, subsample_size=subsample_size,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('eve2_', include_metric=True)

# """Metric EVE OPT ROUND 2"""
# nb_epochs = 27
# lr = 0.047
# bs = 1
# for i in range(1, 26):
#     cur_emb_dim = np.random.randint(8, 65)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = np.random.randint(1, 10001) / 10
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='m_eve2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_eve2_', include_metric=True)

# """METRIC EVE OPT ROUND 3"""
# nb_epochs = 27
# lr = 0.047
# bs = 1
# emb_dim = 63
# margin = 0.3
# scaling = 800
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='m_eve3_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# # ana.score('m_eve3_', include_metric=True)


# """Metric EVE OPT ROUND 4.0"""
# bs = 1
# nb_epochs = 27
# emb_dim = 63
# margin = 0.3
# subsample_size = 4
# scaling = 800
# for lr in (0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1):
#     cur_lr = lr
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', cur_lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_eve4_' + str(cur_lr),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=cur_lr,  scaling=scaling,
#                   batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# # ana.score('m_eve4_', include_metric=True)


# """ADAM OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10000) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='adam_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()


# """ADAM OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(150, 251)
#     cur_lr = np.random.randint(1, 1000) / 100000.
#     cur_bs = int(np.ceil(i / 25))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='adam_1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs),
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """ADAM OPT ROUND 2"""
# nb_epochs = 250
# lr = 0.001
# bs = 10
# for i in range(1, 26):
#     cur_emb_dim = np.random.randint(8, 65)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = np.random.randint(1, 10001) / 10
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='adam2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """ADAM OPT ROUND 3"""
# nb_epochs = 250
# lr = 0.001
# bs = 10
# emb_dim = 64
# margin = 0.6
# scaling = 25
# for t in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='adam3.3_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()

# """Metric ADAM OPT ROUND 2"""
# nb_epochs = 42
# lr = 0.1
# bs = 1
# for i in range(1, 26):
#     cur_emb_dim = np.random.randint(8, 65)
#     cur_margin = np.random.randint(1, 10) / 10
#     cur_scaling = np.random.randint(1, 10001) / 10
#     print('Embedding Dim: ', cur_emb_dim, 'Margin: ', cur_margin, 'Scaling: ', cur_scaling, 'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='m_adam2_' + str(cur_emb_dim) + '_' + str(cur_margin) + '_' + str(cur_scaling),
#                   embedding_dim=cur_emb_dim, margin=cur_margin, scaling=cur_scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=False,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_adam2_', include_metric=True)

# """METRIC ADAM OPT ROUND 3"""
# nb_epochs = 42
# lr = 0.1
# bs = 1
# emb_dim = 48
# margin = 0.9
# scaling = 415.7
# # (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)
# for t in (18, 20, 22, 24, 26, 28, 30, 32):
#     subsample_size = t ** 2
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin: ', margin, 'Scaling: ', scaling,
#           'Number epochs: ',
#           nb_epochs, 'Learning Rate: ', lr, 'Batch Size: ', bs)
#     set = h.Setup(model_name='m_adam3_' + str(subsample_size),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, scaling=scaling,
#                   nb_epochs=nb_epochs, save_config=True, learning_rate=lr, batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# ana.score('m_adam3_', include_metric=True)


# """Metric ADAM OPT ROUND 4.0"""
# bs = 1
# nb_epochs = 42
# emb_dim = 48
# margin = 0.9
# subsample_size = 16
# scaling = 415.7
# for lr in (0.00001, 0.000025, 0.00005, 0.000075, 0.0001, 0.00025, 0.0005,
#            0.00075, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1):
#     cur_lr = lr
#     print('Subsample Size: ', subsample_size, 'Embedding Dim: ', emb_dim, 'Margin:', margin, 'Learning Rate: ', cur_lr,
#           'Batch Size: ', bs)
#     set = h.Setup(model_name='m_adam4_' + str(cur_lr),
#                   subsample_size=subsample_size, embedding_dim=emb_dim, margin=margin, nb_epochs=nb_epochs,
#                   save_config=True, learning_rate=cur_lr, scaling=scaling,
#                   batch_size=bs, include_background=True,
#                   background_pred=True,
#                   nb_iterations=0, embedding_loss=True)
#     set.main()
# # ana.score('m_adam4_', include_metric=True)


# """METRIC CAIN OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10001) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='m_cain_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs), embedding_dim=32,
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
#                   include_background=True,
#                   background_pred=False, margin=0.5,
#                   nb_iterations=3, kernel_bandwidth=None, step_size=1., embedding_loss=True)
#     set.main()
# ana.score('m_cain_', include_metric=True)

# """METRIC CAIN OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10001) / 100000.
#     cur_bs = 1
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='m_cain1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs), embedding_dim=32,
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
#                   include_background=True,
#                   background_pred=False, margin=0.5,
#                   nb_iterations=3, kernel_bandwidth=None, step_size=1., embedding_loss=True)
#     set.main()
# ana.score('m_cain1.1_', include_metric=True)

# """METRIC ABEL OPT ROUND 1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10001) / 100000.
#     cur_bs = int(np.ceil(i / 10))
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='m_abel_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs), embedding_dim=32,
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
#                   include_background=True,
#                   background_pred=True, scaling=25, margin=0.5,
#                   nb_iterations=3, kernel_bandwidth=None, step_size=1., embedding_loss=True)
#     set.main()
# ana.score('m_abel_', include_metric=True)

# """METRIC ABEL OPT ROUND 1.1"""
# for i in range(1, 101):
#     cur_nb_epochs = np.random.randint(10, 251)
#     cur_lr = np.random.randint(1, 10001) / 100000.
#     cur_bs = 1
#     print('Number epochs: ', cur_nb_epochs, 'Learning Rate: ', cur_lr, 'Batch Size: ', cur_bs)
#     set = h.Setup(model_name='m_abel1.1_' + str(cur_lr) + '_' + str(cur_nb_epochs) + '_' + str(cur_bs), embedding_dim=32,
#                   nb_epochs=cur_nb_epochs, save_config=True, learning_rate=cur_lr, batch_size=cur_bs,
#                   include_background=True,
#                   background_pred=True, scaling=25, margin=0.5,
#                   nb_iterations=3, kernel_bandwidth=None, step_size=1., embedding_loss=True)
#     set.main()
# ana.score('m_abel1.1_', include_metric=True)