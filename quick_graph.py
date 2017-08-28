import utils
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
EPOPCHS=200
#one_out_of_ten_no_LCP_TopQuestions_baseline_audio
#derivatives
#one_out_of_ten_no_LCP_TopQuestions_face_NoDerivatives
#one_out_of_ten_no_LCP_TopQuestions_face_audio
directory = '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio/'

auc_dev_list = []
auc_test_list = []


auc_dev_list = np.genfromtxt(directory+"list_dev_prec_rec_auc",delimiter=",",skip_header=0)
auc_test_list = np.genfromtxt(directory+"list_test_prec_rec_auc",delimiter=",",skip_header=0)

print(auc_dev_list.shape[0])
print(auc_test_list.shape[0])
epoch_count = min(auc_dev_list.shape[0],auc_test_list.shape[0])

auc_dev_list = auc_dev_list[0:epoch_count]
auc_test_list = auc_test_list[0:epoch_count]

epochs = np.arange(epoch_count)

plt.ylim([0.5, 1.00])
plt.plot(epochs, auc_dev_list, color="green")
plt.plot(epochs, auc_test_list, color="red")
plt.xlabel("Epochs")
plt.ylabel("AUC")

#plt.plot([0,EPOPCHS],[0.5,0.5],color = 'blue')

green_patch = mpatches.Patch(color='green',label="Development set")
red_patch = mpatches.Patch(color='red',label="Test set")
#blue_patch = mpatches.Patch(color='blue',label="Baseline")

plt.title('AUC of Precision-recall curve for each Epoch')

plt.legend(handles=[green_patch,red_patch],loc=4)

plt.savefig(directory+"prec_rec_AUC_curve.png")