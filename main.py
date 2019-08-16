import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# import category_encoders as ce

draft_history = pd.read_csv('drafthistorynarrow.csv')

# clean up team names and owners
draft_history.loc[draft_history.team_name == 'Argyle Gargoyles', 'Owner'] = 'Mike'
draft_history.loc[draft_history.team_name == 'Nate\'s Team', 'Owner'] = 'Nate'
draft_history.loc[draft_history.team_name == 'Zeke and Des...', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == 'Turn Down fo...', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == '1.21 JJ Watts', 'Owner'] = 'Lisa'
draft_history.loc[draft_history.team_name == 'MINGO was hi...', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == 'Back on Top!', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == 'I let girls...', 'Owner'] = 'Marc'

draft_history.loc[draft_history.team_name == 'MINGO was hi...', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == 'MINGO was hi...', 'Owner'] = 'Marc'
draft_history.loc[draft_history.team_name == 'MINGO was hi...', 'Owner'] = 'Marc'

target = ['position']
input_cols = ['Owner','Pick Overall']

inputs = draft_history[input_cols]
targets = draft_history[target]
le_owner = LabelEncoder()
inputs['Owner'] = le_owner.fit_transform(inputs['Owner'])

# ce_ord = ce.OrdinalEncoder(cols=['Owner'])

x_train, x_test, y_train, y_test = train_test_split(inputs,targets, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


model = RandomForestClassifier()
model.fit(x_train, y_train.values.ravel())
y_predict = model.predict(x_test)
draft_order = ['Marc','Lisa','Lance','Dave','Jodi','Bill','Nate','Kris','Mike','Austin','Marissa','Joy']
# full_draft = draft_order
for item in reversed(draft_order):
    draft_order.append(item)
    
full_draft = 8*draft_order
print(full_draft, len(full_draft))
print(accuracy_score(y_test.values, y_predict))
test_df = pd.DataFrame(
    {'Owner':le_owner.fit_transform(full_draft), 
    'Overall Pick': list(range(1,193))}
)
# for x in le_owner.inverse_transform(test_df['Owner']):
#     print(x, end=' ')
# print('\n')    
# # print(le_owner.inverse_transform(test_df['Owner']))
# for x in test_df['Overall Pick']:
#     print(x, end=' ')
# print('\n')
draft_results = pd.DataFrame({'Pick':list(range(1,193)),'Owner':full_draft, 'Position':model.predict(test_df)})

def highlight_position(val):
    if val == 'QB':
        color = '#009933'
    elif val == 'RB':
        color = '#6699ff'
    elif val == 'WR':
        color = '#ffcc66'
    elif val == 'TE':
        color = '#cc3300'
    else:
        color = '#d3d3d3'
    
    return 'background-color: %s' % color

draft_results.pivot(index='Round',columns='Owner',values='Position')[draft_order[:12]].style.applymap(highlight_position)
# print()
