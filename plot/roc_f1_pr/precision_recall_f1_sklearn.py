from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, fbeta_score

y_true = [0, 0, 0, 2, 1, 2, 0, 1, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 0, 2, 1, 2]

accuracy_default = accuracy_score(y_true, y_pred) # Return the number of correctly classified samples
accuracy_norm = accuracy_score(y_true, y_pred, normalize=False) # Return the fraction of correctly classified samples
print("accuracy:", accuracy_default, accuracy_norm)

# Calculate precision score
precision_macro = precision_score(y_true, y_pred, average='macro')
precision_micro = precision_score(y_true, y_pred, average='micro')
precision_default = precision_score(y_true, y_pred, average=None)
print("precision:", precision_macro, precision_micro)

# Calculate recall score
recall_macro = recall_score(y_true, y_pred, average='macro')
recall_micro = recall_score(y_true, y_pred, average='micro')
recall_default = recall_score(y_true, y_pred, average=None)
print("recall:", recall_macro, recall_micro, recall_default)

# Calculate f1 score
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_default = f1_score(y_true, y_pred, average=None)
print("f1:", f1_macro, f1_micro, f1_default)

# Calculate f beta score
fb_macro = fbeta_score(y_true, y_pred, average='macro', beta=0.5)
fb_micro = fbeta_score(y_true, y_pred, average='micro', beta=0.5)
fb_default = fbeta_score(y_true, y_pred, average=None, beta=0.5)
print("fb:", fb_macro, fb_micro, fb_default)
