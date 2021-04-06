import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        self.assertEqual(len(self.boolq_dataset),len(self.dataset))

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        test_tensor=torch.zeros([self.max_seq_len])
        for i in range(len(self.boolq_dataset)):
            ith=self.boolq_dataset[i]
            ith_input=ith.get('input_ids')
            ith_mask=ith.get('attention_mask')
            ith_label=ith.get('labels')
            self.assertEqual(ith_input.shape,test_tensor.shape)
            self.assertEqual(ith_mask.shape,test_tensor.shape)
            self.assertEqual(ith_input.dtype,torch.long)
            self.assertEqual(ith_mask.dtype,torch.long)
            print(type(ith_input))
            print(type(ith_mask))
            print(type(ith_label))
            if ith_label == 1:
                return True
            elif ith_label == 0:
                return True
            else:
                break

            
        
        


if __name__ == "__main__":
    unittest.main()
