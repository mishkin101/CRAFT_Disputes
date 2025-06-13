# def finetune_craft(config):
#     globals().update(config)
#     #handle logic for loading data:
#     utterance_metadata = finetune_utterance_metadata 
#     conversation_metadata = finetune_convo_metadata
#     loaded_corpus = loadDataset()
#     #get conversations and utterances dataframe:
#     convo_dataframe = loaded_corpus.get_conversations_dataframe()
#     utterance_dataframe = loaded_corpus.get_utterances_dataframe()
#     #load device:
#     device = loadDevice()

#     #create training logic:
#     X_train_id, X_test_id, y_train_id, y_test_id = createTrainTestSplit(convo_dataframe)
#     convo_dataframe_main = assignSplit(convo_dataframe, train_ids=X_train_id, test_ids=X_test_id)
#     X_train = convo_dataframe.loc[X_train_id]
#     X_test = convo_dataframe.loc[X_test_id]
#     #same splits for each k-fold index
#     train_val_id_list = createTrainValSplit(X_train)
#     # Collect the full validation‐score history from each fold (not just the single best model).
#     # Average those fold histories epoch-wise to find which epoch maximizes the mean validation score across folds.
#     all_folds =[]
#     for fold, pair in enumerate(train_val_id_list, start=1):
#         fold_metrics =[]
#         model_dir, train_dir, results_dir, plots_dir, config_dir = build_fold_directories(fold)
#         convo_dataframe_fold = assignSplit(convo_dataframe, train_ids=pair[0], val_ids=pair[1])
#         train_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_train, split_key="train")
#         val_pairs = loadLabeledPairs(voc, utterance_dataframe, convo_dataframe_fold, last_only = last_only_val, split_key="val")
#         #create epoch iterations:
#         epoch_iters = computerIterations(train_pairs)
#         total_loss = 0
#         for epoch in range(finetune_epochs +1, start =1):
#             batch_metrics = trainEpoch(train_pairs, craft_model, epoch_iters, voc, total_loss, epoch)
#             epoch_metrics = evalEpoch(val_pairs, craft_model, total_loss, epoch)
#             fold_metrics.append(epoch_metrics)
#             #save training results for each fold:
#             save_experiment_results_train_batch(train_dir, batch_metrics)
#             save_experiment_results_train_epoch(train_dir, epoch_metrics)
#     all_folds.append(fold_metrics)
#     mean_results = average_across_folds(all_folds)
#     if ray_tune:
#             report_dict = { f"mean_{k}": v for k,v in mean_results.items() }
#             tune.report(epoch=epoch, **report_dict)
#             mlflow.log_metrics(report_dict, step=epoch)
#     save_avg_metrics(experiment_dir, mean_results)