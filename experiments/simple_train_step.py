from itertools import chain
import torch


def simple_train_step(optimizer, criterion, device, model, split_val):

    def train_step(x, y):
        model.to(device)
        premise, hypothesis = torch.split(x, split_val, dim=1)

        first_hidden_pre = torch.rand_like(premise).to(device)
        decoder_input_pre = torch.rand_like(premise).to(device)
        first_hidden_hyp = torch.rand_like(hypothesis).to(device)
        decoder_input_hyp = torch.rand_like(hypothesis).to(device)

        y_hat = model(
            premise,
            hypothesis,
            first_hidden_pre,
            decoder_input_pre,
            first_hidden_hyp,
            decoder_input_hyp,
        )
        # print(y_hat)
        # print(torch.nn.functional.one_hot(y.long(),num_classes=3).squeeze(0))
        loss = criterion(
            y_hat,
            torch.nn.functional.one_hot(y.long(), num_classes=3).squeeze(0).float(),
        ).float()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return train_step
