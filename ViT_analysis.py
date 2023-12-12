# ViT-B/16
byte_per_para = 4
B = 64 # batch size
P = 16*16 # patch size
# P = 16*16 # patch size
C = 1000 # num of classes
IMAGE = 224*224 # image size
T = IMAGE/P # sequence length
L = 12 # layers
N = 12 # heads
E = 768 # embedding
bytes_to_mbs = 1_000_000

''' Model Parameters = Patch_embedding + Attention + MLP_classifier '''
# Single Attention layer #
QKVLiner = 4*E*E
MLP = 2*(E*(4*E))
Layer_Norm = 2*E
Attention_Single = QKVLiner+MLP+Layer_Norm
# Single Attention layer #
Attention = L*Attention_Single
Patch_embedding = P*E
MLP_classifier = E*C
Model_paras = Patch_embedding + MLP_classifier + Attention
print('The overall Model Paras is(MB) ',Model_paras*byte_per_para/bytes_to_mbs,'with Attention',Attention*byte_per_para/bytes_to_mbs,'(',Attention/Model_paras*100,'%)')

'''Act Parameters = Patch_embedding + Attention + MLP_classifier'''
# Single Attention Layer #
Residual = 2*E*T
Layer_Norm = 2*E*T
Softmax = N*T*T
QKVLiner = 4*T*E
MLP = 6*T*E
Attention_Single = Residual+Layer_Norm+Softmax+QKVLiner+MLP
# Single Attention Layer #
Attention = L*Attention_Single
Patch_embedding = IMAGE + E*T
MLP_classifier = C
Act_paras = B*(Patch_embedding + MLP_classifier + Attention)
print('The overall Act Paras is(MB) per batch',Act_paras*byte_per_para/bytes_to_mbs,'with Attention',B*Attention*byte_per_para/bytes_to_mbs,'(',B*Attention/Act_paras*100,'%)')

print('The overall Training Mem is(MB) ',(3*Model_paras+2*Act_paras)*byte_per_para/bytes_to_mbs,
      'with Model',(3*Model_paras)*byte_per_para/bytes_to_mbs,'(',3*Model_paras/(3*Model_paras+2*Act_paras)*100,'%)'
      ' with Act',(2*Act_paras)*byte_per_para/bytes_to_mbs,'(',2*Act_paras/(3*Model_paras+2*Act_paras)*100,'%)')


'''Act Prompt Parameters =  Model + QKV '''
# Single Attention Layer #
QKV_single = 3*T*E
# Single Attention Layer #
QKV = L*QKV_single
Act_paras = B*(QKV)
print('The Prompt Tuning Mem is(MB) ',(1*Model_paras+1*Act_paras)*byte_per_para/bytes_to_mbs,
      'with Model',(1*Model_paras)*byte_per_para/bytes_to_mbs,'(',1*Model_paras/(1*Model_paras+1*Act_paras)*100,'%)'
      ' with Act',(1*Act_paras)*byte_per_para/bytes_to_mbs,'(',1*Act_paras/(1*Model_paras+1*Act_paras)*100,'%)')
