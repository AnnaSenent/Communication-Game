
import argparse

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
        "--mode",
        type=str,
        default="gs",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: gs)",
    )

    parser.add_argument(
        "--N", type=int, default=20, help="0 to N range for the input integers"
    )

    parser.add_argument(
        "--n_integers", type=int, default=2, help="Number of integers to sum"
    )

    parser.add_argument(
        "--validation_batch_size", type=int, default=0, help="Batch size for the validation data"
    )
    # Specify arguments used during the training process

    # This is only for gs mode
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Temperature for the sender agent in gs mode (default : 1.0)"
    )

    parser.add_argument(
        "--sender_hidden", type=int, default=16, help="Size of the hidden layer for the sender agent (default : 16)"
    )

    parser.add_argument(
        "--receiver_hidden", type=int, default=16, help="Size of the hidden layer for the receiver agent (default : 16)"
    )

    # Specify argument to control the output
    parser.add_argument(
        "--print_validation_events",
        default=False, action="store_true",
        help="Print the validation data, the messages produced by the sender andthe output probabilities produced by the receiver"
    )

    args = core.init(parser, params)

    return args


def load_data(params):

    args = get_params(params)
    data = SumDataset(args.train, args.N, args.n_integers)

    train_data = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_data = DataLoader(SumDataset(args.val, args.N, args.n_integers), batch_size=args.validation_batch_size, shuffle=False, num_workers=1)

    n_features = data.get_n_features()

    return train_data, val_data, n_features


def main(params):

    args = get_params(params)
    train_data, val_data, n_features = load_data(params)
    sender = SumSender(n_features=n_features, n_hidden=args.sender_hidden)
    receiver = SumReceiver(n_features=n_features, n_hidden=args.receiver_hidden)


    def loss(_sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):

        acc = (receiver_output.argmax(dim=0) == labels).detach().float()

        loss = F.cross_entropy(receiver_output, labels, reduction='none')

        return loss, {'acc': acc}

    if args.mode.lower() == 'gs':
        agent_sender = core.GumbelSoftmaxWrapper(sender, temperature=args.temperature)
        agent_receiver = core.SymbolReceiverWrapper(receiver, vocab_size=args.vocab_size,
                                                agent_input_size=args.receiver_hidden)

        game = core.SymbolGameGS(agent_sender, agent_receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=agent_sender, decay=0.9, minimum=0.1)]

    else:
        agent_sender = core.ReinforceWrapper(sender)
        agent_receiver = core.SymbolReceiverWrapper(receiver, vocab_size=args.vocab_size,
                                                    agent_input_size=args.receiver_hidden)
        agent_receiver = core.ReinforceDeterministicWrapper(agent_receiver)

        game = core.SymbolGameReinforce(agent_sender, agent_receiver, loss, sender_entropy_coeff=0.05,
                                        receiver_entropy_coeff=0.0)
        callbacks = []

    optimizer = core.build_optimizer(game.parameters())

    if args.print_validation_events == True:

        callbacks.extend([core.ConsoleLogger(print_train_loss=True, as_json=True),
                     core.PrintValidationEvents(n_epochs=args.n_epochs)])

        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_data,
                               validation_data=val_data, callbacks=callbacks)

    else:

        callbacks.extend([core.ConsoleLogger(print_train_loss=True, as_json=True)])

        trainer = core.Trainer(game=game, optimizer=optimizer, train_data=train_data,
                               validation_data=val_data, callbacks=callbacks)

    trainer.train(n_epochs=args.n_epochs)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])

