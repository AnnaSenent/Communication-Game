

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_readers import SumDataset
from architectures import SumSender, SumReceiver

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents


def get_params(params):

    parser = argparse.ArgumentParser(description="Play the summation game")

    # We specify our parameters here

    # Specify the path to the train and test data
    parser.add_argument(
        "--train", type=str, help="Path to the train data"
    )

    parser.add_argument(
        "--val", type=str, help="Path to the validation data"
    )

    # Add features
    parser.add_argument(
        "--N", type=int, default=20, help="0 to N range for the input integers"
    )

    parser.add_argument(
        "--n_integers", type=int, default=2, help="Number of integers to sum"
    )

    parser.add_argument(
        "--validation_batch_size", type=int, default=0, help="Batch size for the validation data"
    )
    # Specify arguments used during the training process (only for gs mode)
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for the sender agent in gs mode (default : 1.0)"
    )

    # Specify arguments to define the agents
    parser.add_argument(
        "--sender_cell", type=str, default="rnn", help="Cell used for the sender agent {rnn, gru, lstm} (default : rnn)"
    )

    parser.add_argument(
        "--receiver_cell", type=str, default="rnn", help="Cell used for the receiver agent {rnn, gru, lstm} (default : rnn)"
    )

    parser.add_argument(
        "--sender_hidden", type=int, default=16, help="Size of the hidden layer for the sender agent (default : 16)"
    )

    parser.add_argument(
        "--receiver_hidden", type=int, default=16, help="Size of the hidden layer for the receiver agent (default : 16)"
    )

    parser.add_argument(
            "--sender_embedding", type=int, default=16, help="Output dimensionality for the layer that embeds symbols produced by the sender (default : 16)"
    )

    parser.add_argument(
        "--receiver_embedding", type=int, default=16, help="Output dimensionality for the layer that embeds messages for the receiver (default : 16)"
    )

    # Specify argument to control the output
    parser.add_argument(
        "--print_validation_events",
        default=False, action="store_true",
        help="Print the validation data, the messages produced by the sender andthe output probabilities produced by the receiver"
    )

    # args = parser.parse_args()
    args = core.init(parser, params)

    return args

def load_data(params):

    args = get_params(params)

    data = SumDataset(args.train, args.N, args.n_integers)

    train_data = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # train_data = DataLoader(data.get_dataset(), batch_size=args.batch_size, shuffle=True, num_workers=1)
    train_labels = DataLoader(data.get_labels(), batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_data = DataLoader(SumDataset(args.val, args.N, args.n_integers).get_dataset(), batch_size=args.validation_batch_size, shuffle=False, num_workers=1)
    val_labels = DataLoader(SumDataset(args.val, args.N, args.n_integers).get_labels(), batch_size=args.validation_batch_size, shuffle=False, num_workers=1)

    n_features = data.get_n_features()

    return train_data, train_labels, val_data, val_labels, n_features


def egg_gs_mode(params):

    args = get_params(params)
    train_data, train_labels, val_data, val_labels, n_features = load_data(params)
    sender = SumSender(n_features=n_features, n_hidden=args.sender_hidden)
    receiver = SumReceiver(n_features=n_features, n_hidden=args.receiver_hidden)

    def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        acc = (receiver_output.argmax(dim=1) == labels).detach().float()

        loss = F.cross_entropy(receiver_output, labels, reduction='none')

        return loss, {'acc': acc}

    agent_sender = core.RnnSenderGS(
        sender,
        vocab_size=args.vocab_size,
        embed_dim=args.sender_embedding,
        hidden_size=args.sender_hidden,
        cell=args.sender_cell,
        max_len=args.max_len,
        temperature=args.temperature
    )

    agent_receiver = core.RnnReceiverGS(
        receiver,
        vocab_size=args.vocab_size,
        embed_dim=args.receiver_embedding,
        hidden_size=args.receiver_hidden,
        cell=args.receiver_cell
    )

    game = core.SenderReceiverRnnGS(agent_sender, agent_receiver, loss)



    optimizer = core.build_optimizer(game.parameters())




    if args.print_validation_events == True:

        callbacks = [core.TemperatureUpdater(agent=agent_sender, decay=0.9, minimum=0.1),
                     core.ConsoleLogger(print_train_loss=True, as_json=True),
                     core.PrintValidationEvents(n_epochs=args.n_epochs)]

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_data,
            validation_data=val_data,
            callbacks=callbacks)

    else:

        callbacks = [core.TemperatureUpdater(agent=agent_sender, decay=0.9, minimum=0.1),
                     core.ConsoleLogger(print_train_loss=True, as_json=True)]

        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_data,
            validation_data=val_data,
            callbacks=callbacks)

    trainer.train(n_epochs=args.n_epochs)


if __name__ == "__main__":
    import sys

    egg_gs_mode(sys.argv[1:])

