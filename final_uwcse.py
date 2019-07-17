from  Lib.ILPRLEngine import *
import argparse
from Lib.mylibw import read_by_tokens
from Lib.DNF import DNF
from Lib.CNF import CNF
from Lib.CONJ  import CONJ
from Lib.cgt import CGT


from Lib.PredicateLibV5 import PredFunc
from sklearn.metrics import accuracy_score ,precision_recall_curve,auc,precision_recall_fscore_support,average_precision_score,log_loss
from sklearn.metrics import  roc_auc_score ,precision_recall_curve,auc,precision_recall_fscore_support,accuracy_score,confusion_matrix
import pandas as pd
import csv
import operator
import scipy.signal


# for 5-fold we should run the program 5 times with TEST_SET_INDEX from 0 to 4
parser = argparse.ArgumentParser()

# parser.add_argument('--RATIO_POS',default=3.0,help='Batch Size',type=float)
parser.add_argument('--TEST_SET_INDEX',default=4,help='0-4 the index of the 5-fold experiment',type=int)
parser.add_argument('--CHECK_CONVERGENCE',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--SHOW_PRED_DETAILS',default=0,help='Print predicates definition details',type=int)
parser.add_argument('--PRINTPRED',default=0,help='Print predicates',type=int)
parser.add_argument('--SYNC',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--L2LOSS',default=0,help='Use L2 instead of cross entropy',type=int)
parser.add_argument('--BS',default=1,help='Batch Size',type=int)
parser.add_argument('--T',default=4,help='Number of forward chain',type=int)
parser.add_argument('--LR_SC', default={ (-1000,2):.005 ,  (2,1e5):.01} , help='Learning rate schedule',type=dict)
parser.add_argument('--ITEM_REMOVE_ITER',default=10000 ,help='length period of each item removal',type=int)
parser.add_argument('--MAXTERMS',default=6 ,help='Maximum number of terms in each clause',type=int)
parser.add_argument('--L1',default=0 ,help='Penalty for maxterm',type=float)
parser.add_argument('--L2',default=.0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--L3',default=0 ,help='Penalty for distance from binary',type=float)
parser.add_argument('--ALLTIMESTAMP',default=0 ,help='Add loss for each timestamp',type=int)
parser.add_argument('--LR', default=.003 , help='Base learning rate',type=float)
parser.add_argument('--FILT_TH_MEAN', default=1 , help='Fast convergence total loss threshold MEAN',type=float)
parser.add_argument('--FILT_TH_MAX', default=1 , help='Fast convergence total loss threshold MAX',type=float)
parser.add_argument('--OPT_TH', default=1, help='Per value accuracy threshold',type=float)
parser.add_argument('--PLOGENT', default=.350 , help='Crossentropy coefficient',type=float)
parser.add_argument('--BETA1', default=.90 , help='ADAM Beta1',type=float)
parser.add_argument('--BETA2', default=.999 , help='ADAM Beta2',type=float)
parser.add_argument('--EPS', default=1e-6, help='ADAM Epsillon',type=float)
parser.add_argument('--GPU', default=1, help='Use GPU',type=int)
parser.add_argument('--ITER', default=100000, help='Maximum number of iteration',type=int)
parser.add_argument('--ITER2', default=10, help='Epoch',type=int)
parser.add_argument('--LOGDIR', default='./logs/Logic', help='Log Dir',type=str)
parser.add_argument('--TB', default=0, help='Use Tensorboard',type=int)
parser.add_argument('--ADDGRAPH', default=1, help='Add graph to Tensorboard',type=int)
parser.add_argument('--CLIP_NORM', default=0, help='Clip gradient',type=float)

parser.add_argument('--PRINT_WEIGHTS',default=0,help='Print raw weights',type=int)
parser.add_argument('--SEED',default=0,help='Random seed',type=int)
parser.add_argument('--BINARAIZE', default=0 , help='Enable binrizing at fast convergence',type=int)
parser.add_argument('--MAX_DISP_ITEMS', default=50 , help='Max number  of facts to display',type=int)
parser.add_argument('--W_DISP_TH', default=.1 , help='Display Threshold for weights',type=int)
parser.add_argument('--DISP_BATCH_VALUES',default=[],help='Batch Size',type=list)
args = parser.parse_args()

print('displaying config setting...')
for arg in vars(args):
        print( '{}-{}'.format ( arg, getattr(args, arg) ) )
 

################################################

Classes={}
dics=[{} for _ in range(5) ]

names=[ 'language.db' , 'graphics.db', 'systems.db','theory.db','ai.db']

for i in range(5):

    DATA_FILES_I = './data/uwcse/rl/' + names[i]
    with open(DATA_FILES_I, 'r')   as datafile:
        line = datafile.readline()
        while line:
            line=line.replace('(',',')
            line=line.replace(')','')
            line=line.replace('\n','')
            cols = line.split(',')
            cols= [c.strip() for c in cols]
            if cols[0] in dics[i]:
                dics[i][cols[0]].append( cols[1:] )
            else:
                dics[i][cols[0]] = [cols[1:]]
            line = datafile.readline()


#extract the constants from the facts and exaples in the datafiles
################################################

C_person=[ [] for i in range(5) ]
C_course= [ [] for i in range(5) ]
C_quarter=[ [] for i in range(5) ]
C_titles=[ [] for i in range(5) ]
C_projects=[ [] for i in range(5) ]

C_phase=[ 'Pre_Quals', 'Post_Quals', 'Post_Generals']
C_level=[ 'Level_300' , 'Level_400', 'Level_500']
C_position= [ 'Faculty', 'Faculty_adjunct','Faculty_affiliate','Faculty_emeritus' ]


cnt=0

for i in range(5):
    for item in dics[i]['professor']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])

    for item in dics[i]['student']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        
    for item in dics[i]['publication']:
        if item[0] not in C_titles[i]:
            C_titles[i].append(item[0])
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])

    for item in dics[i]['taughtBy']:
        
        if item[0] not in C_course[i]:
            C_course[i].append(item[0])
        
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])
        
        if item[2] not in C_quarter[i]:
            C_quarter[i].append(item[2])

    for item in dics[i]['courseLevel']:
        if item[0] not in C_course[i]:
            C_course[i].append(item[0])    

        if item[1] not in C_level:
            C_level.append(item[1])    



    for item in dics[i]['hasPosition']:

        
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        
        if item[1] not in C_position:
            C_position.append(item[1])

    if 'projectMember' in dics[i]:
        for item in dics[i]['projectMember']:
            if item[1] not in C_person[i]:
                C_person[i].append(item[1])
            
            if item[0] not in C_projects[i]:
                C_projects[i].append(item[0])

    for item in dics[i]['advisedBy']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])
        
        cnt+=1


    for item in dics[i]['inPhase']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        if item[1] not in C_phase:
            C_phase.append(item[1])
    
    
    for item in dics[i]['tempAdvisedBy']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])

    for item in dics[i]['yearsInProgram']:
        if item[0] not in C_person[i]:
            C_person[i].append(item[0])
        

    for item in dics[i]['ta']:
        
        if item[0] not in C_course[i]:
            C_course[i].append(item[0])
        
        if item[1] not in C_person[i]:
            C_person[i].append(item[1])
        
        if item[2] not in C_quarter[i]:
            C_quarter[i].append(item[2])

    

# print(cnt)
# exit(0)
    

    
    # for item in dics[i]['introCourse']:
    #     if item[0] not in C_course[i]:
    #         C_course[i].append(item[0])

   
    
# we use unified names to be able to generalize across 5 experiments   
#   {p_1,...,p_{nPerson} }   for people names , ....
################################################

nPerson = np.max( [len(C_person[i]) for i in range(5)])
nTitles = np.max( [len(C_titles[i]) for i in range(5)])
nCourse = np.max( [len(C_course[i]) for i in range(5)])
nQuarters = np.max( [len(C_quarter[i]) for i in range(5)])
nProjects = np.max( [len(C_projects[i]) for i in range(5)])


Persons=[ 'p_%d'%(i+1) for i in range(nPerson)]    
Titles=[ 't_%d'%(i+1) for i in range(nTitles)]    
Courses=[ 'c_%d'%(i+1) for i in range(nCourse)]    
Quaters=[ 'q_%d'%(i+1) for i in range(nQuarters)]    
Projects=[ 'pr_%d'%(i+1) for i in range(nProjects)]    


def PI( item,i):
    return Persons[ C_person[i].index(item)] 
def PT( item,i):
    return Titles[ C_titles[i].index(item)] 
def CI( item,i):
    return Courses[ C_course[i].index(item)] 
def QI( item,i):
    return Quaters[ C_quarter[i].index(item)] 

def XI( item,i):
    return Projects[ C_projects[i].index(item)] 

pair_tempAdvisedBye=[]
for i in range(5):
    for item in dics[i]['tempAdvisedBy']:
        pair_tempAdvisedBye.append (  (PI(item[0],i),  PI(item[1],i) ) )

pair_publication=[]
for i in range(5):
    for item in dics[i]['publication']:
        pair_publication.append (  (PI(item[1],i),  PT(item[0],i) ) )


#defining the predicate functions
################################################
        
Constants = { 'P':Persons, 'C':Courses, 'Q':Quaters, 'T':Titles, 'X':Projects}
predColl = PredCollection (Constants)
 


# predColl.add_pred(dname='student',arguments=['P'])
# predColl.add_pred(dname='professor',arguments=['P'])

# predColl.add_pred(dname='professor',arguments=['P'], variables=['P'], pFunc = DNF('professor',terms=1,init=[1,-2,-1,.1],sig=1)  , use_neg=True, Fam='or',chunk_count=0)

# predColl.add_pred(dname='samePerson',arguments=['P','P'] )

predColl.add_pred(dname='publication',arguments=['P','T'] ,pairs=pair_publication)
predColl.add_pred(dname='has_publication',arguments=['P'] )
predColl.add_pred(dname='taughtby_course',arguments=['P', 'C'] )
predColl.add_pred(dname='ta_course',arguments=['P', 'C'] )
predColl.add_pred(dname='taughtby_quarter',arguments=['P', 'Q'] )
predColl.add_pred(dname='ta_quarter',arguments=['P', 'Q'] )
predColl.add_pred(dname='tempAdvisedBy',arguments=['P','P'],pairs=pair_tempAdvisedBye )


# for level in C_level:
#     predColl.add_pred(dname='courseLevel_%s'%level,arguments=['C'] )

# predColl.add_pred(dname='introCourse',arguments=['C'] )

for pos in C_position:
    predColl.add_pred(dname='hasPosition_%s'%pos,arguments=['P'] )


# predColl.add_pred(dname='hasanyPosition',arguments=['P'] )

predColl.add_pred(dname='projectMember',arguments=['P','X'] )


for phase in C_phase:
    predColl.add_pred(dname='inPhase_%s'%phase,arguments=['P'] )
predColl.add_pred(dname='inanyPhase',arguments=['P'] )


predColl.add_pred(dname='student',arguments=['P'], variables=['P'], pFunc = DNF('student',terms=2,predColl=predColl, init_terms=['advisedBy(A,B)','tempAdvisedBy(A,B)'],fast=True)  , use_neg=False, Fam='or',chunk_count=0,inc_preds=['advisedBy','tempAdvisedBy'])
predColl.add_pred(dname='professor',arguments=['P'], variables=['P'], pFunc = DNF('professor',terms=2,predColl=predColl, init_terms=['advisedBy(B,A)','tempAdvisedBy(B,A)'],fast=True)  , use_neg=False, Fam='or',chunk_count=0,inc_preds=['advisedBy','tempAdvisedBy'])


N_Y = 2
for i in range(N_Y):
    predColl.add_pred(dname='year_%d'%i,arguments=['P'] ,pFunc = CGT(name='year_%d'%i, init_w=i/10.0,c=30.0) , Fam='eq', max_T=1,inc_preds=['year_%d'%i])

# N_TA =4
# for i in range(N_TA):
#     predColl.add_pred(dname='ta_count_%d'%i,arguments=['P'] )

# N_TU =4
# for i in range(N_TU):
#     predColl.add_pred(dname='taught_count_%d'%i ,arguments=['P'])

# N_PA = 4
# for i in range(N_PA):
#     predColl.add_pred(dname='publication_count_%d'%i,arguments=['P'])


# predColl.add_pred(dname='sameQ',arguments=['P','P'], variables=['Q'] ,pFunc = 
#     CONJ('sameQ',init=[-1,.1],sig=2,init_terms=['ta_quarter(A,C), taughtby_quarter(B,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['taughtby_quarter','ta_quarter'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='sameC',arguments=['P','P'], variables=['C'] ,pFunc = 
#     CONJ('sameC',init=[-1,.1],sig=2,init_terms=['ta_course(A,C), taughtby_course(B,C)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['taughtby_course','ta_course'], exc_conds=[ ],  Fam='or',chunk_count=0) 


predColl.add_pred(dname='sameProj',arguments=['P','P'], variables=['X'] ,pFunc = 
    DNF('sameProj',terms=1,init=[1,.1,-1,.1],sig=1,init_terms=['projectMember(A,C), projectMember(B,C)'],predColl=predColl,fast=True)  , use_neg=True, inc_preds=['projectMember','samePerson'], exc_conds=[ ],  Fam='or',chunk_count=0) 

predColl.add_pred(dname='colaborate',arguments=['P','P'], variables=['T'] ,pFunc = 
    DNF('colaborate',terms=1,init=[1,.1,-1,.1],sig=1,init_terms=['publication(A,C), publication(B,C)'],predColl=predColl,fast=True)  , use_neg=True, inc_preds=['publication','samePerson'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='colaborateStudent',arguments=['P','P'], variables=[] ,pFunc = 
#     DNF('colaborateStudent',terms=1,init=[1,.1,-1,.1],sig=1,init_terms=['colaborate(A,B), student(A), student(B)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['colaborate','student'], exc_conds=[ ],  Fam='or',chunk_count=0) 

# predColl.add_pred(dname='colaborateProf',arguments=['P','P'], variables=[] ,pFunc = 
#     DNF('colaborateProf',terms=1,init=[1,.1,-1,.1],sig=1,init_terms=['colaborate(A,B), professor(A), professor(B)'],predColl=predColl,fast=True)  , use_neg=False, inc_preds=['colaborate','professor'], exc_conds=[ ],  Fam='or',chunk_count=0) 

ai=0
 

incs=['advisedBy']
for i in range(10):
    incs.append('aux%d'%i)

 
 
predColl.add_pred(dname='advisedBy1',oname='advisedBy', arguments=['P','P'] , variables=[  'C','Q' ] ,pFunc = DNF('advisedBy1',terms=1,init=[-1,.1,-1,.1],sig=2)  , use_neg=True , Fam='or',chunk_count=nPerson)
predColl.add_pred(dname='advisedBy2',oname='advisedBy', arguments=['P','P'] , variables=[  'T' ] ,pFunc = DNF('advisedBy2',terms=1,init=[-1,.1,-1,.1],sig=2)  , use_neg=True , Fam='max',chunk_count=nTitles)
# predColl.add_pred(dname='advisedBy3',oname='advisedBy', arguments=['P','P'] , variables=[   ] ,pFunc = DNF('advisedBy3',terms=1,init=[-1,.1,-1,.1],sig=2)  , use_neg=True , Fam='or',chunk_count=0)
 
    

predColl.initialize_predicates()    






# defining 5 background knowledge structures corresponding to 5 experiments

all_bgs=[] 

for i in range(5):
    bg = Background( predColl ) 
    
    try:
        for p in Persons:
            bg.add_backgroud( 'samePerson', (p,p) )

    except:
            print(i,'same person')

    for item in dics[i]['student']:
        try:
            bg.add_backgroud( 'student', ( PI(item[0],i) ,) )
        except:
            print(i,item,'student')

    for item in dics[i]['professor']:
        try:
            bg.add_backgroud( 'professor', ( PI(item[0],i) ,) )
        except:
            print(i,item,'professor')

    for item in dics[i]['publication']:
        try:
            bg.add_backgroud( 'has_publication', ( PI(item[1],i) ,) )
        except:
            print(i,item,'publication')
        try:
            bg.add_backgroud( 'publication', ( PI(item[1],i) ,PT(item[0],i)) )
        except:
            print(i,item,'publication 2')

    
    for item in dics[i]['taughtBy']:
        try:
            bg.add_backgroud( 'taughtby_course', ( PI(item[1],i), CI(item[0],i) ) )
        except:
            print(i,item,'taughtby 1')
        try:
            bg.add_backgroud( 'taughtby_quarter', ( PI(item[1],i), QI(item[2],i) ) )
        except:
            print(i,item,'taughtby 2')
        # try:
        #     bg.add_backgroud( 'professor', ( PI(item[1],i) ,) )
        # except:
        #     print(i,item,'taughtby 3')
    
    for item in dics[i]['courseLevel']:
        try:
            bg.add_backgroud( 'courseLevel_'+item[1], ( CI(item[0],i) ,) )
        except:
            print(i,item,'courseLevel 1')


    for item in dics[i]['hasPosition']:
        
        try:
            bg.add_backgroud( 'hasanyPosition', ( PI(item[0],i) ,) )
        except:
            print(i,item,'hasPosition 0')
        
        try:
            bg.add_backgroud( 'hasPosition_'+item[1], ( PI(item[0],i) ,) )
        except:
            print(i,item,'hasPosition 1')
        # try:
        #     bg.add_backgroud( 'professor', ( PI(item[0],i) ,) )
        # except:
        #     print(i,item,'hasPosition 2')


    if 'projectMember' in dics[i]:
        for item in dics[i]['projectMember']:
            try:
                bg.add_backgroud( 'projectMember', ( PI(item[1],i) ,XI(item[0],i)) )
            except:
                print(i,item,'projectMember 1')



    for item in dics[i]['inPhase']:

        try:
            bg.add_backgroud( 'inanyPhase', ( PI(item[0],i) ,) )
        except:
            print(i,item,'inPhase 0')

        try:
            bg.add_backgroud( 'inPhase_'+item[1], ( PI(item[0],i) ,) )
        except:
            print(i,item,'inPhase 1')
        # try:
        #     bg.add_backgroud( 'student', ( PI(item[0],i) ,) )
        # except:
        #     print(i,item,'inPhase 2')

    

    
    for item in dics[i]['tempAdvisedBy']:
        try:
            bg.add_backgroud( 'tempAdvisedBy', ( PI(item[0],i), PI(item[1],i) ) )
        except:
            print(i,item,'tempadviseby 1')


    for item in dics[i]['yearsInProgram']:

        parts=item[1].split('_')
        year = int( parts[1] )

        try:
            for n in range(N_Y):
                bg.add_backgroud( 'year_%d'%n, ( PI(item[0],i) ,), value=year/10. )
        except:
            print(i,item,'yearsInProgram 1')




    for item in dics[i]['ta']:
        try:
            bg.add_backgroud( 'ta_course', ( PI(item[1],i), CI(item[0],i) ) )
        except:
            print(i,item,'ta 1')
        try:
            bg.add_backgroud( 'ta_quarter', ( PI(item[1],i), QI(item[2],i) ) )
        except:
            print(i,item,'ta 2')
       
    

    

    
    for item in dics[i]['advisedBy']:
        try:
            bg.add_example( 'advisedBy', ( PI(item[0],i), PI(item[1],i) ) , value=1.0)
        except:
            print(i,item,'advisedBy 1')

     
    # adding all the negative examples for the target predicate
    
    bg.add_all_neg_example('advisedBy')    
    # if i==args.TEST_SET_INDEX:
    #     bg.add_all_neg_example('advisedBy')    
    # else:
    #     bg.add_all_neg_example_ratio('advisedBy',50)
    

    # for p in Persons:
    #     cnt=0
    #     for item in bg.backgrounds['ta_quarter']:
    #         if p==item[0]:
    #             cnt+=1
    #     if cnt>0:
    #         for n in range(N_TA):
    #             bg.add_backgroud( 'ta_count_%d'%n , (p,) , float(cnt>n))

    #     cnt=0   
    #     for item in bg.backgrounds['taughtby_quarter']:
    #         if p==item[0]:
    #             cnt+=1
    #     if cnt>0:
    #         for n in range(N_TU):
    #             bg.add_backgroud( 'taught_count_%d'%n , (p,) ,float(cnt>n))
        
        
    #     cnt=0
    #     for item in bg.backgrounds['publication']:
    #         if p==item[0]:
    #             cnt+=1
    #     if cnt>0:
    #         for n in range(N_PA):
    #             bg.add_backgroud( 'publication_count_%d'%n , (p,) , float(cnt>n) )
    
    all_bgs.append(bg)
    
     
 

# this is the callback function that provides the training algorithm with different background knowledge for training and testing
###############################################
def bgs(it,is_train):
     
    if is_train:
        #excluding the background corresponding to the test data
         
        inds = [ i for i in range(5)  if i != args.TEST_SET_INDEX]
        # inds = [ i for i in range(5)  if i != args.TEST_SET_INDEX]
        # return [ all_bgs[i] for i in  inds ]
        # index = np.random.randint(4)
        return [ all_bgs[ inds[it%4] ] ]
    else:
        return [ all_bgs[args.TEST_SET_INDEX] ]

    
    
# this callback function is called for custom display each time a testing background is gonna be tested
# ###########################################################################

def disp_fn(eng,it,session,cost,outp):
    
    mismatch_count = 0

    bg=all_bgs[args.TEST_SET_INDEX]

    true_class = bg.get_target_data('advisedBy')
    pred_class =outp['advisedBy'][0,:]

    
    
    avg_acc = average_precision_score(true_class, pred_class)
    auroc = roc_auc_score(true_class, pred_class)
    cll = np.mean( true_class*np.log(pred_class+1e-7) + (1.0-true_class)*np.log(1.0-pred_class+1e-7) )
    x,y,th1 = precision_recall_curve(true_class, pred_class)
    aupr = auc(y,x)

    cll2 = log_loss(true_class, pred_class)

    acc = accuracy_score(true_class, (pred_class>=.5).astype(np.float) )
    
    print('--------------------------------------------------------')    
    print('acc : %.4f , avg acc : %4f, AUROC : %.4f,  AUPR : %.4f, cll : %.4f, cll2 : %.4f'% (acc,avg_acc,auroc,aupr,cll,cll2) )

    # for th in [.1,.2,.3,.4,.5,.6,.7,.8]:
    #     pred_class = outp['advisedBy'][0,:]>th
    #     pred_class = pred_class.astype(np.float)

    #     print(th, 'Classifdication error Count %d  of total %d '%(np.sum(true_class!=pred_class),len(true_class)) )
    #     pred=outp['advisedBy'][0,:]
    #     acc = accuracy_score(true_class, pred_class)
    #     auroc = roc_auc_score(true_class, pred_class)
    #     x,y,th1 = precision_recall_curve(true_class, pred_class)
    #     aupr = auc(y,x)
    #     print(th,'accuracy score : %.4f , AUROC score : %.4f,  AUPR score : %.4f'% (acc,auroc,aupr) )

    return

     

model = ILPRLEngine( args=args ,predColl=predColl ,bgs=bgs ,disp_fn=disp_fn)
model.train_model()    

