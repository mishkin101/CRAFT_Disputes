
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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



# def get_default_config():
#     # Assuming config.py defines these variables:
#     default_cfg = {
#         "dropout": dropout,
#         "batch_size": batch_size,
#         "clip": clip,
#         "learning_rate": learning_rate,
#         "print_every": print_every,
#         "finetune_epochs": finetune_epochs,
#         "validation_size": val_size,
#         "ray_tune": True,  # enable reporting inside finetune_craft
#         "optimizer_type": optimizer_type,
#         "loss_function": loss_function,
#         "activation": activation,
#         "k_folds": k_folds,
#         "last_only_train": last_only_train,
#         "last_only_val": last_only_val,
#         "corpus_name": corpus_name,
#         "finetune_data": finetune_data,
#         utt_label_metadata,
#         label_metadata,
#         pretrained_model,
#         pretrain_exclude_phrases,
#         pretrain_case


#         # add any other custom config variables from config.py
#     }
#     return default_cfg


def plot_split_ratios_from_df(
    convo_df,
    split_col="meta.split",
    outcome_col="meta.provided_outcome",
    preferred_splits=("train","val","test")
):

    # 1) get arrays
    splits = convo_df[split_col].astype(str)
    labels = convo_df[outcome_col].astype(int)

    # 2) determine split categories in preferred order, then any others
    all_splits = splits.unique().tolist()
    cats = [s for s in preferred_splits if s in all_splits] \
         + sorted(s for s in all_splits if s not in preferred_splits)
    n = len(cats)

    # 3) count successes/impasses per split
    success = np.zeros(n, int)
    impasse = np.zeros(n, int)
    for i, sp in enumerate(cats):
        sub = labels[splits == sp]
        success[i] = (sub == 0).sum()
        impasse[i] = (sub == 1).sum()
    total = success + impasse

    # avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        p_success = np.where(total>0, success/total, 0.0)
        p_impasse = np.where(total>0, impasse/total, 0.0)

    # 4) plot
    x = np.arange(n)
    width = 0.6

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x, p_success, width, label="Success (0)", color="tab:green")
    ax.bar(x, p_impasse, width, bottom=p_success, label="Impasse (1)", color="tab:red")

    # annotate
    for i in x:
        ax.text(i, p_success[i]/2, f"{p_success[i]*100:.0f}%",
                ha="center", va="center", color="white", fontsize=10)
        ax.text(i, p_success[i] + p_impasse[i]/2, f"{p_impasse[i]*100:.0f}%",
                ha="center", va="center", color="white", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Split")
    ax.set_title("Split‐wise Success vs. Impasse Ratios")
    ax.legend(loc="center")
    plt.tight_layout()
    plt.show()


def plot_counts_and_avg_lengths_from_df(
    convo_df,
    split_col="meta.split",
    outcome_col="meta.provided_outcome",
    length_col="meta.convo_len",
    preferred_splits=("train","val","test")
):

    # 1) Determine split categories in preferred order, then any extras
    all_splits = convo_df[split_col].dropna().unique().tolist()
    splits = [s for s in preferred_splits if s in all_splits] \
           + sorted(s for s in all_splits if s not in preferred_splits)
    n_splits = len(splits)

    # 2) Prepare containers
    labels = [0,1]  # 0=Success, 1=Impasse
    counts = np.zeros((2, n_splits), int)      # row=label, col=split
    avglen = np.zeros((2, n_splits), float)

    # 3) Fill counts & avglen
    for li, lab in enumerate(labels):
        sub = convo_df[convo_df[outcome_col] == lab]
        grp = sub.groupby(split_col)[length_col]
        for ci, sp in enumerate(splits):
            if sp in grp.groups:
                group = grp.get_group(sp)
                counts[li,ci] = len(group)
                avglen[li,ci] = group.mean()
            else:
                counts[li,ci] = 0
                avglen[li,ci] = 0.0

    # 4) Compute proportions for each split
    total_counts = counts.sum(axis=0)
    # avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        p_success = np.where(total_counts>0, counts[0] / total_counts, 0.0)
        p_impasse = np.where(total_counts>0, counts[1] / total_counts, 0.0)

    # 5) Plotting
    x = np.arange(n_splits)
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,8), sharex=True)
    # annotate
    for i in x:
        ax1.text(i, p_success[i]/2, f"{p_success[i]*100:.0f}%",
                ha="center", va="center", color="white", fontsize=10)
        ax1.text(i, p_success[i] + p_impasse[i]/2, f"{p_impasse[i]*100:.0f}%",
                ha="center", va="center", color="white", fontsize=10)
    # Top: stacked proportions
    ax1.bar(x, p_success, width, label="Success (0)", color="tab:green")
    ax1.bar(x, p_impasse, width, bottom=p_success, label="Impasse (1)", color="tab:red")
    ax1.set_ylabel("Proportion")
    ax1.set_title("Proportion of Outcomes by Split")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="center")
    # Bottom: avg lengths
    width2 = 0.25
    bars_succ = ax2.bar(x- width2/2, avglen[0], width2, color="tab:green", label="Success (0)")
    bars_imp  = ax2.bar(x + width2/2, avglen[1], width2, color="tab:red",   label="Impasse (1)")
    ax2.set_ylabel("Avg. Convo Length")
    ax2.set_title("Average Conversation Length by Split & Outcome")
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.legend(loc="center")
    # annotate
    for bar in bars_succ + bars_imp:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                h / 2,
                f"{h:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10
            )

    plt.tight_layout()
    plt.show()



def plot_fold_summary(
    data_folds,
    split_col="meta.split",
    outcome_col="meta.provided_outcome",
    length_col="meta.convo_len",
    preferred_splits=("train","val","test")
):
    all_splits = set()
    for df in data_folds:
        all_splits |= set(df[split_col].dropna().unique())
    splits = [s for s in preferred_splits if s in all_splits] \
           + sorted(s for s in all_splits if s not in preferred_splits)
    n_splits = len(splits)
    n_folds  = len(data_folds)
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    # 2) Compute metrics arrays
    p_succ   = np.zeros((n_folds, n_splits))
    p_imp    = np.zeros((n_folds, n_splits))
    avg_succ = np.zeros((n_folds, n_splits))
    avg_imp  = np.zeros((n_folds, n_splits))

    for i, df in enumerate(data_folds):
        for j, split in enumerate(splits):
            sub = df[df[split_col] == split]
            n0 = (sub[outcome_col] == 0).sum()
            n1 = (sub[outcome_col] == 1).sum()
            tot = n0 + n1
            if tot > 0:
                p_succ[i,j]   = n0 / tot
                p_imp[i,j]    = n1 / tot
            if n0 > 0:
                avg_succ[i,j] = sub[sub[outcome_col]==0][length_col].mean()
            if n1 > 0:
                avg_imp[i,j]  = sub[sub[outcome_col]==1][length_col].mean()
    cmap = plt.get_cmap("tab10")
    split_colors = {split: cmap(k) for k, split in enumerate(splits)}
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(14,8), sharex=True)
    x = np.arange(n_folds)
    cluster_w = 0.8
    bar_w     = cluster_w / n_splits
    for j, split in enumerate(splits):
        xs = x - cluster_w/2 + bar_w*(j + 0.5)
        ax1.bar(xs,      p_succ[:,j], bar_w,
                color=split_colors[split], label=split)
        ax1.bar(xs,      p_imp[:,j], bar_w,
                bottom=p_succ[:,j],
                color=split_colors[split], hatch="///")
        for i, xi in enumerate(xs):
            if p_succ[i,j] > 0:
                ax1.text(xi, p_succ[i,j]/2,
                         f"{p_succ[i,j]*100:.0f}%",
                         ha="center", va="center",
                         color="white", fontsize=10)
            if p_imp[i,j] > 0:
                ax1.text(xi, p_succ[i,j] + p_imp[i,j]/2,
                         f"{p_imp[i,j]*100:.0f}%",
                         ha="center", va="center",
                         color="white", fontsize=10)
    ax1.set_title("Outcome Proportions by Split & Fold")
    ax1.set_ylabel("Proportion")
    ax1.set_ylim(0,1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels)

    # --- Bottom: avg conversation lengths ---
    sub_w = bar_w / 2
    for j, split in enumerate(splits):
        xs = x - cluster_w/2 + bar_w*(j + 0.5)
        ax2.bar(xs - sub_w/2, avg_succ[:,j], sub_w,
                color=split_colors[split], label=split)
        ax2.bar(xs + sub_w/2, avg_imp[:,j], sub_w,
                color=split_colors[split], hatch="///")
        for i, xi in enumerate(xs):
            if avg_succ[i,j] > 0:
                ax2.text(xi - sub_w/2, avg_succ[i,j]/2,
                         f"{avg_succ[i,j]:.1f}",
                         ha="center", va="center",
                         color="white", fontsize=10)
            if avg_imp[i,j] > 0:
                ax2.text(xi + sub_w/2, avg_imp[i,j]/2,
                         f"{avg_imp[i,j]:.1f}",
                         ha="center", va="center",
                         color="white", fontsize=10)

    ax2.set_title("Average Conversation Length by Split & Fold")
    ax2.set_ylabel("Avg. Convo Length")
    ax2.set_xlabel("Fold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(fold_labels)

    # --- Combined legend on right ---
    # split handles (solid)
    split_handles = [
        Patch(facecolor=split_colors[s], edgecolor="black", label=s)
        for s in splits
    ]
    # outcome handles
    outcome_handles = [
        Patch(facecolor="white", edgecolor="black", label="Success", hatch=""),
        Patch(facecolor="white", edgecolor="black", label="Impasse", hatch="///"),
    ]

    fig.legend(
        handles=split_handles + outcome_handles,
        labels=splits + ["Success","Impasse"],
        title="Splits & Outcomes",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5)
    )

    plt.tight_layout(rect=[0,0,0.85,1])
    plt.show()




def plot_fold_summary_with_ai(
    data_folds,
    split_col="meta.split",
    outcome_col="meta.provided_outcome",
    ai_cols=("meta.buyer_is_AI", "meta.seller_is_AI"),
    length_col="meta.convo_len",
    preferred_splits=("train", "val", "test")
):
    # 1) determine splits & folds
    all_splits = set()
    for df in data_folds:
        all_splits |= set(df[split_col].dropna().unique())
    splits = [s for s in preferred_splits if s in all_splits] + sorted(all_splits - set(preferred_splits))
    n_splits = len(splits)
    n_folds  = len(data_folds)
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    # 2) init arrays
    p_succ   = np.zeros((n_folds, n_splits))
    p_imp    = np.zeros((n_folds, n_splits))
    avg_succ = np.zeros((n_folds, n_splits))
    avg_imp  = np.zeros((n_folds, n_splits))
    p_ai_succ = np.zeros((n_folds, n_splits))
    p_ai_imp  = np.zeros((n_folds, n_splits))

    # 3) compute metrics
    for i, df in enumerate(data_folds):
        df = df.copy()
        df["contains_AI"] = df[ai_cols[0]] | df[ai_cols[1]]

        for j, split in enumerate(splits):
            sub = df[df[split_col] == split]
            n0 = (sub[outcome_col] == 0).sum()
            n1 = (sub[outcome_col] == 1).sum()
            tot = n0 + n1
            if tot:
                p_succ[i,j] = n0 / tot
                p_imp[i,j]  = n1 / tot
            if n0:
                avg_succ[i,j] = sub.loc[sub[outcome_col]==0, length_col].mean()
                p_ai_succ[i,j] = sub.loc[sub[outcome_col]==0, "contains_AI"].mean()
            if n1:
                avg_imp[i,j]  = sub.loc[sub[outcome_col]==1, length_col].mean()
                p_ai_imp[i,j]  = sub.loc[sub[outcome_col]==1, "contains_AI"].mean()

    # 4) colors & layout
    cmap = plt.get_cmap("tab10")
    split_colors = {split: cmap(k) for k, split in enumerate(splits)}
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    x = np.arange(n_folds)
    cluster_w = 0.8
    bar_w     = cluster_w / n_splits

    # --- Top plot: outcome proportions ---
    ax = axes[0]
    for j, split in enumerate(splits):
        xs = x - cluster_w/2 + bar_w*(j + 0.5)
        s = p_succ[:,j];  i_ = p_imp[:,j]
        ax.bar(xs,      s, bar_w, color=split_colors[split])
        ax.bar(xs,      i_, bar_w, bottom=s, color=split_colors[split], hatch="///")

        # annotate
        for xi, ss, ii in zip(xs, s, i_):
            ax.text(xi, ss/2,       f"{ss*100:.0f}%", ha="center", va="center", color="white", fontsize=9)
            ax.text(xi, ss+ii/2,    f"{ii*100:.0f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_title("Outcome Proportions by Split & Fold", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_ylim(0,1)

    # legend
    handles = []
    for split in splits:
        handles.append(Patch(facecolor=split_colors[split], label=split))
    handles += [
        Patch(facecolor="white", edgecolor="black", label="Success", hatch=""),
        Patch(facecolor="white", edgecolor="black", label="Impasse", hatch="///"),
    ]
    ax.legend(handles=handles, title="Splits & Outcomes",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    # --- Middle: avg lengths ---
    ax = axes[1]
    sub_w = bar_w/2
    for j, split in enumerate(splits):
        xs = x - cluster_w/2 + bar_w*(j + 0.5)
        s_len = avg_succ[:,j];  i_len = avg_imp[:,j]
        ax.bar(xs - sub_w/2, s_len, sub_w, color=split_colors[split])
        ax.bar(xs + sub_w/2, i_len, sub_w, color=split_colors[split], hatch="///")

        for xi, ss, ii in zip(xs, s_len, i_len):
            if ss>0: ax.text(xi-sub_w/2, ss/2, f"{ss:.1f}", ha="center", va="center", color="white", fontsize=9)
            if ii>0: ax.text(xi+sub_w/2, ii/2, f"{ii:.1f}", ha="center", va="center", color="white", fontsize=9)

    ax.set_title("Average Conversation Length by Split & Fold", fontsize=14)
    ax.set_ylabel("Avg. Convo Length", fontsize=12)

    # legend (reuse split handles)
    ax.legend(handles=[Patch(facecolor=split_colors[s]) for s in splits],
              labels=splits, title="Split",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    # --- Bottom: AI proportions within outcome ---
    ax = axes[2]
    for j, split in enumerate(splits):
        xs = x - cluster_w/2 + bar_w*(j + 0.5)
        s_ai = p_ai_succ[:,j];  i_ai = p_ai_imp[:,j]
        # success side
        ax.bar(xs - sub_w/2,       1-s_ai, sub_w, color=split_colors[split])
        ax.bar(xs - sub_w/2,       s_ai,   sub_w, bottom=1-s_ai, color="gray")
        # impasse side
        ax.bar(xs + sub_w/2,       1-i_ai, sub_w, color=split_colors[split], hatch="///")
        ax.bar(xs + sub_w/2,       i_ai,   sub_w, bottom=1-i_ai, color="black", hatch="///")

        for xi, sa, ia in zip(xs, s_ai, i_ai):
            ax.text(xi-sub_w/2, 1-sa/2, f"{sa*100:.0f}%", ha="center", va="center", color="white", fontsize=9)
            ax.text(xi+sub_w/2, 1-ia/2, f"{ia*100:.0f}%", ha="center", va="center", color="white", fontsize=9)

    ax.set_title("AI-Involving Conversations by Outcome, Split & Fold", fontsize=14)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)

    # legend
    ai_handles = [
        Patch(facecolor="gray", edgecolor="black", label="Success & AI"),
        Patch(facecolor="black", edgecolor="black", label="Impasse & AI", hatch="///"),
    ]
    ax.legend(handles=[Patch(facecolor=split_colors[s]) for s in splits] + ai_handles,
              labels=list(splits) + ["Success & AI", "Impasse & AI"],
              title="Split / AI-Outcome",
              bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    plt.tight_layout()
    plt.show()
