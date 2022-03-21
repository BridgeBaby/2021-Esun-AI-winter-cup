import pandas as pd
import numpy as np
import torch


class CardDataset(data.Dataset):
    def __init__(self, file, test=False):
        self.data_frame = pd.read_csv(file)
        self.test = test
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        data = self.data_frame.iloc[idx]
        user_id = data.user_id
        
        shop_history_tag = eval(data.sequence_shop_tag)
        shop_history_amount = eval(data.sequence_amount)
        shop_history_count = eval(data.sequence_count)
        shop_history_month = eval(data.sequence_month)
        # shop_history_card = np.array(data.sequence_card, dtype=object)
        # shop_history_pt = np.array(data.sequence_pt, dtype=object)
        shop_history_card = np.asarray(eval(data.sequence_card), dtype=float)
        shop_history_pt = np.asarray(eval(data.sequence_pt), dtype=float)

        if self.test:
            target_seprate = None
            target_percentage = np.zeros(len(to_predict))
        else:
            target_seprate = list(shop_history_month).index(shop_history_month[-1])

            target_tag = torch.LongTensor(shop_history_tag[target_seprate:])
            target_amount = torch.LongTensor(shop_history_amount[target_seprate:])
            # target_count = torch.LongTensor(shop_history_count[target_seprate:])
            # target_month = torch.LongTensor(shop_history_month[target_seprate:])
            target_dict = {tag: amount for tag, amount in zip(target_tag, target_amount) if tag in to_predict}
            target_percentage = np.zeros(len(to_predict))
            if target_dict:
                for tag, amount in target_dict.items():
    #                 if math.isnan(amount / sum(target_dict.values())):
    #                     print(tag, amount, sum(target_dict.values()))
                    target_percentage[to_predict.index(tag)] = amount / sum(target_dict.values()) if not math.isnan(amount / sum(target_dict.values())) else 0
        shop_history_tag = torch.LongTensor(shop_history_tag[:target_seprate])
        shop_history_amount = torch.FloatTensor(shop_history_amount[:target_seprate]).div(2000)
        shop_history_count = torch.FloatTensor(shop_history_count[:target_seprate])
        shop_history_month = torch.LongTensor(shop_history_month[:target_seprate])
        shop_history_card = torch.FloatTensor(shop_history_card[:target_seprate])
        shop_history_pt = torch.FloatTensor(shop_history_pt[:target_seprate])
        
        marriage = data.marriage
        education = data.education
        occupation = data.occupation
        country = data.country
        workas = data.position
        source = data.source
        quota = data.quota
        gender = data.gender
        age = data.age
        marraige_age_gender = data.marraige_age_gender
        return user_id, shop_history_tag, shop_history_amount, target_percentage, shop_history_count, shop_history_month, shop_history_card, shop_history_pt, marriage, education, occupation, country, workas, source, quota, gender, age, marraige_age_gender

def my_collate(batch):
    """
    padding shop history
    """
    # batch contains a list of tuples of structure (sequence, target)
    data_list = []
    for i in range(len(batch[0])):
        try:
            data = [item[i] for item in batch]
        except IndexError:
            data = [item for item in batch]
        if i in [1, 2, 4, 5]:
            data = pad_sequence(data, padding_value=0, batch_first=True)
        if i in [6, 7]:
            data = pad_sequence(data, padding_value=0, batch_first=True)
        if isinstance(data, list):
            try:
                data = torch.tensor(np.array(data))
            except:
                data = torch.cat(data, dim=0)
        data_list.append(data.detach())
    return data_list
