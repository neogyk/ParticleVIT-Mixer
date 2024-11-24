import numpy as np


# Get median model sorted by auc
def get_median_bg_reject(
    input_True_labels, input_taggers_dict, tag_eff=0.3, remove_outliers=False
):

    # ---------------------------------------
    # Get tag efficiency working point
    if tag_eff == 0.5:
        point = 225  # 0.05 + 225*(1 - 0.05)/476=0.5
    elif tag_eff == 0.3:
        point = 125  # 0.05 + 125*(1 - 0.05)/476=0.3
    elif tag_eff == 0.4:
        point = 175

    base_tpr = np.linspace(0.05, 1, 476)
    input_taggers_median = []

    all_bg_reject = []
    all_bg_reject_std = []

    all_bg_reject_outliers = []
    all_bg_reject_std_outliers = []

    all_aucs = []
    all_auc_std = []
    all_aucs_outliers = []
    all_auc_std_outliers = []

    len_models = []

    for k in range(len(input_taggers_dict["out_prob"])):

        out_probs = input_taggers_dict["out_prob"][k]

        bg_reject = []
        bg_reject_std = []
        aucs = []

        for i in range(len(out_probs)):

            auc = roc_auc_score(input_True_labels, out_probs[i])
            aucs.append(auc)

            fpr, tpr, thresholds = roc_curve(
                input_True_labels, out_probs[i], pos_label=1, drop_intermediate=False
            )

            interp_fpr = interp(base_tpr, tpr, fpr)

            bg_reject.append(1.0 / interp_fpr[point])

        all_bg_reject.append(np.median(bg_reject))
        all_bg_reject_std.append(np.std(bg_reject))

        all_aucs.append(np.median(aucs))
        all_auc_std.append(np.std(aucs))

        if remove_outliers:
            scores = np.asarray(bg_reject)

            p25 = np.percentile(scores, 1 / 6.0 * 100.0)
            p75 = np.percentile(scores, 5 / 6.0 * 100)

            # Get mean and std for the bg rejection
            robust_mean = np.mean(
                [scores[i] for i in range(len(scores)) if p25 <= scores[i] <= p75]
            )
            robust_std = np.std(
                [scores[i] for i in range(len(scores)) if p25 <= scores[i] <= p75]
            )

            indices = [
                i
                for i in range(len(scores))
                if robust_mean - 3 * robust_std
                <= scores[i]
                <= robust_mean + 3 * robust_std
            ]

            new_scores = scores[indices]

            len_models.append(len(new_scores))

            all_bg_reject_outliers.append(np.median(new_scores))
            all_bg_reject_std_outliers.append(np.std(new_scores))

            new_aucs = np.asarray(aucs)[indices]
            all_aucs_outliers.append(np.median(new_aucs))
            all_auc_std_outliers.append(np.std(new_aucs))

    if remove_outliers:

        return (
            len_models,
            all_bg_reject_outliers,
            all_bg_reject_std_outliers,
            all_aucs_outliers,
            all_auc_std_outliers,
        )

    else:

        return all_bg_reject, all_bg_reject_std, all_aucs, all_auc_std
