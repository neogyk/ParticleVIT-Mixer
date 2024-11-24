import pdb
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from background_rej import get_median_bg_reject

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset")
parser.add_argument("-type", "--type")

args = parser.parse_args()


df = pd.read_csv(args.dataset)
df["pred_label"] = df["prediction"].apply(lambda x: 1 if x > 0.5 else 0)

roc_auc = roc_auc_score(df["pred_label"], df["labels"])
acc = accuracy_score(df["pred_label"], df["labels"])

fpr, tpr, threshold = roc_curve(df["labels"], df["prediction"])
# get_median_bg_reject
pdb.set_trace()

print(f"ROC AUC {roc_auc}")
print(f"ACC : {acc}")
print(f"FPR TPR :{fpr}, {tpr}")

# ROC AUC PLOT:
import matplotlib.pyplot as plt

plt.title("Receiver Operating Characteristic")
plt.plot(tpr, fpr)  # , 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
# plt.show()
plt.savefig(f"roc_auc_{args.type}.png")
