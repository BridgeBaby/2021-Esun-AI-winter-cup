import math
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from loss import LambdaLoss, NDCG
from module.preprocess import CardDataset, my_collate


class MultiHotBST(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        
        self.embeddings_tag = nn.Embedding(
                int(N_UNIQUE['shop_tag'])+1, int(math.sqrt(N_UNIQUE['shop_tag']))+1
                )
        self.embeddings_position = nn.Embedding(
               sequence_length, 15
            )
        self.embeddings_card = nn.Embedding(
            int(N_UNIQUE['card'])+1, int(math.sqrt(N_UNIQUE['card']))+1
        )
        self.embeddings_pt = nn.Embedding(
            int(N_UNIQUE['pt'])+1, int(math.sqrt(N_UNIQUE['pt']))+1
        )

        self.embeddings_marriage = nn.Embedding(
            N_UNIQUE['masts'], int(math.sqrt(N_UNIQUE['masts']))
        )
        self.embeddings_education = nn.Embedding(
            N_UNIQUE['educd'], int(math.sqrt(N_UNIQUE['educd']))
        )
        self.embeddings_occupation = nn.Embedding(
            N_UNIQUE['trdtp'], int(math.sqrt(N_UNIQUE['trdtp']))
        )
        self.embeddings_country = nn.Embedding(
            N_UNIQUE['naty'], int(math.sqrt(N_UNIQUE['naty']))
        )
        self.embeddings_workas = nn.Embedding(
            N_UNIQUE['poscd'], int(math.sqrt(N_UNIQUE['poscd']))
        )
        self.embeddings_source = nn.Embedding(
            N_UNIQUE['cuorg'], int(math.sqrt(N_UNIQUE['cuorg']))
        )
        self.embeddings_gender = nn.Embedding(
            N_UNIQUE['gender_code'], int(math.sqrt(N_UNIQUE['gender_code']))
        )
        self.embeddings_age = nn.Embedding(
            N_UNIQUE['age'], int(math.sqrt(N_UNIQUE['age']))
        )
        self.embeddings_quota = nn.Embedding(
            45, int(math.sqrt(45))
        )
        self.embeddings_mixed = nn.Embedding(
            47, int(math.sqrt(46))+1
        )
        
        # Network
        self.transformerlayer_0 = nn.TransformerEncoderLayer(15, 3, dropout=0.1)
        self.transformerlayer_1 = nn.TransformerEncoderLayer(15, 3, dropout=0.1)
        self.transformerlayer_2 = nn.TransformerEncoderLayer(15, 3, dropout=0.1)
        self.linear = nn.Sequential(
            nn.Linear(
                208,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 16),
            # nn.Softmax(dim=1)
        )
        self.criterion = LambdaLoss()
        self.ndcg = NDCGLoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
    
    def multi_hot_embedding(self, tags, card, pt, weights, positions):
        batch_size = len(tags)
        position_length = len(torch.unique(positions)) - 1
        current_positions = torch.tensor([], device=self.device)
        embedd_matrix_card = self.embeddings_card(torch.arange(1, 16, device=self.device)).repeat(batch_size, 1, 1)
        embedd_matrix_pt = self.embeddings_pt(torch.arange(1, 5, device=self.device)).repeat(batch_size, 1, 1)
        embedd_tag = self.embeddings_tag(tags)
        embedd_card = torch.bmm(card, embedd_matrix_card)
        embedd_pt = torch.bmm(pt, embedd_matrix_pt)
        for p in torch.unique(positions):
            if p == 0:
                continue
            current_position = (positions == p).unsqueeze(dim=1)
            current_positions = torch.cat((current_positions, current_position), dim=1)
        
        pos_mul_weight = torch.mul(current_positions, weights.unsqueeze(dim=1).repeat(1, position_length, 1)).float()
        activate_positions = (torch.any((current_positions), dim=2))
        mask = (activate_positions==1).unsqueeze(dim=2)
        
        # tag
        sequence_tensor = pos_mul_weight.bmm(embedd_tag)
        mhe_tag = torch.masked_select(sequence_tensor, mask).reshape((batch_size, 12, 8))
        
        # card
        sequence_tensor = pos_mul_weight.bmm(embedd_card)
        mhe_card = torch.masked_select(sequence_tensor, mask).reshape((batch_size, 12, 4))
        
        # pt
        sequence_tensor = pos_mul_weight.bmm(embedd_pt)
        mhe_pt = torch.masked_select(sequence_tensor, mask).reshape((batch_size, 12, 3))
        
        return mhe_tag, mhe_card, mhe_pt                                                                
                                                                     
    def encode_input(self, inputs):
        (user_id, shop_history_tag, shop_history_amount, target_percentage, shop_history_count,
         shop_history_month, shop_history_card, shop_history_pt, marriage, education, occupation,
         country, workas, source, quota, gender, age, marraige_age_gender = inputs)

        shop_history_tag, shop_history_card, shop_history_pt = self.multi_hot_embedding(shop_history_tag,
                                                                                        shop_history_card,
                                                                                        shop_history_pt,
                                                                                        shop_history_amount,
                                                                                        shop_history_month)
        user_sequence_features = torch.cat((shop_history_tag, shop_history_card, shop_history_pt), dim=2)
        # del shop_history_tag, shop_history_card, shop_history_pt
        positions = torch.arange(0, sequence_length, 1, dtype=int, device=self.device)
        positions = self.embeddings_position(positions)
        user_sequence_features = user_sequence_features + positions

        # user_id = self.embeddings_user_id(user_id)
        marriage = self.embeddings_marriage(marriage)
        education = self.embeddings_education(education)
        occupation = self.embeddings_occupation(occupation)
        country = self.embeddings_country(country)
        workas = self.embeddings_workas(workas)
        source = self.embeddings_source(source)
        gender = self.embeddings_gender(gender)
        age = self.embeddings_age(age)
        quota = self.embeddings_quota(quota)
        marraige_age_gender = self.embeddings_mixed(marraige_age_gender)
        user_features = torch.cat((marriage, education, occupation, country, workas, source, gender, age, quota), 1)

        return user_sequence_features, user_features, target_percentage

    def forward(self, batch):
        user_sequence_features, user_features, target_rating = self.encode_input(batch)
        transformer_output = self.transformerlayer_0(user_sequence_features)
        transformer_output = self.transformerlayer_1(transformer_output)
        transformer_output = self.transformerlayer_2(transformer_output)
        transformer_output = torch.flatten(transformer_output, start_dim=1)
        features = torch.cat((transformer_output, user_features), dim=1)
        output = self.linear(features.float())
        return output, target_rating

    def training_step(self, batch, batch_idx):
        out, target_rating, = self(batch)
        loss = self.criterion(out, target_rating)

        mae = self.mae(out, target_rating)
        mse = self.mse(out, target_rating)
        rmse = torch.sqrt(mse)
        self.log(
            "train_mae", mae, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log(
            "train_rmse", rmse, on_step=True, on_epoch=False, prog_bar=False
        )

        self.log("train_step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        out, target_rating = self(batch)
        loss = self.criterion(out, target_rating)
        ndcg = self.ndcg(out, target_rating)
    
        mae = self.mae(out, target_rating)
        mse = self.mse(out, target_rating)
        rmse = torch.sqrt(mse)
        
        return {"ndcg": ndcg.detach(), "loss": loss, "mae": mae.detach(), "rmse":rmse.detach()}
    
    def test_step(self, batch, batch_idx,):
        out, target_rating = self(batch)
        top3_indices = torch.topk(out, 3, dim=1).indices.cpu().numpy().tolist()
        return top3_indices
            
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()
        avg_ndcg = torch.stack([x["ndcg"] for x in outputs]).mean()
        print(f'''\n
***validation NDCG = {avg_ndcg:.5f}***
        ''')
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_mae", avg_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_ndcg", avg_ndcg, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs):
        # print(len(outputs))
        # y_hat = torch.cat([x for x in outputs])
        # y_hat = y_hat.tolist()
        dfs = []
        for out in outputs:
            top1 = [predit_dict[x[0]] for x in out]
            top2 = [predit_dict[x[1]] for x in out]
            top3 = [predit_dict[x[2]] for x in out]
            data = {"top1": top1, "top2": top2, "top3": top3}
            dfs.append(pd.DataFrame.from_dict(data))
        result = pd.concat(dfs)
        result.to_csv("lightning_logs/predict.csv", index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00001)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.00001)
        return parser
    
    def setup(self, stage=None):
        print("Loading datasets")
        self.train_dataset = CardDataset("data/12m_v4/train_data.csv")
        self.val_dataset = CardDataset("data/12m_v4/valid_data.csv")
        # self.test_dataset = CardDataset("data/12m_new_amt/valid_data.csv", test=True)
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=512,
            shuffle=True,
            num_workers=5,
            collate_fn=my_collate,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=512,
            shuffle=False,
            num_workers=5,
            collate_fn=my_collate,
            pin_memory=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=5,
        )
