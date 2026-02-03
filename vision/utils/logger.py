import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import copy
import json


class Logger:
    def __init__(self, opt, pretrain=True):
        self.opt = opt

        if opt.validate:
            self.val_loss = [[] for i in range(opt.model_splits)]
        else:
            self.val_loss = None

        self.train_loss = [[] for i in range(opt.model_splits)] if pretrain else [[]]

        self.other_loss = {'loss_neg':[[] for i in range(opt.model_splits)] if pretrain else [[]],
                           'loss_aux':[[] for i in range(opt.model_splits)] if pretrain else [[]]}

        if opt.start_epoch > 0:
            self.loss_last_training = np.load(
                os.path.join(opt.model_path, "train_loss.npy")
            ).tolist()
            self.train_loss[:len(self.loss_last_training)] = copy.deepcopy(self.loss_last_training)
            print(self.train_loss)

            if opt.validate:
                self.val_loss_last_training = np.load(
                    os.path.join(opt.model_path, "val_loss.npy")
                ).tolist()
                self.val_loss[:len(self.val_loss_last_training)] = copy.deepcopy(self.val_loss_last_training)
            else:
                self.val_loss = None
        else:
            self.loss_last_training = None

            if opt.validate:
                self.val_loss = [[] for i in range(opt.model_splits)] if pretrain else [[]]
            else:
                self.val_loss = None

        self.num_models_to_keep = 1
        assert self.num_models_to_keep > 0, "Dont delete all models!!!"
        self.l1_save_freq = opt.save_freq

    def create_log(
        self,
        model,
        accuracy=None,
        epoch=0,
        optimizer=None,
        final_test=False,
        final_loss=None,
        acc5=None,
        classification_model=None,
        log_module_idx=None
    ):

        print("Saving model and log-file to " + self.opt.log_path)

        # Save the model checkpoint
        if classification_model is None:
            if self.opt.experiment == "vision":
                for idx, layer in enumerate(model.module.encoder):
                    if (log_module_idx is None) or (log_module_idx == idx): # if only saving certain module, all module saved in epoch 0
                        torch.save(
                            layer.state_dict(),
                            os.path.join(self.opt.log_path, "model_{}_{}.ckpt".format(idx, epoch)),
                        )
                torch.save(
                            model.module.loss.state_dict(),
                            os.path.join(self.opt.log_path, "loss_{}.ckpt".format(epoch)),
                        )
            else:
                torch.save(
                    model.state_dict(),
                    os.path.join(self.opt.log_path, "model_{}.ckpt".format(epoch)),
                )

            ### remove old model files to keep dir uncluttered
            if (epoch + 1 - self.num_models_to_keep) % self.opt.save_freq != 0:
                try:
                    if self.opt.experiment == "vision":
                        for idx, _ in enumerate(model.module.encoder):
                            if idx == 0 and (epoch + 1 - self.num_models_to_keep) % self.l1_save_freq == 0:
                                continue
                            if log_module_idx is None or (log_module_idx == idx): # if only saving certain module
                                print("Deleting old model file {}".format(
                                    os.path.join(
                                        self.opt.log_path,
                                        "model_{}_{}.ckpt".format(idx, epoch - self.num_models_to_keep),
                                    )
                                ))
                                os.remove(
                                    os.path.join(
                                        self.opt.log_path,
                                        "model_{}_{}.ckpt".format(idx, epoch - self.num_models_to_keep),
                                    )
                                )
                        os.remove(
                            os.path.join(
                                self.opt.log_path,
                                "loss_{}.ckpt".format(epoch - self.num_models_to_keep),
                            )
                        )
                    else:
                        os.remove(
                            os.path.join(
                                self.opt.log_path,
                                "model_{}.ckpt".format(epoch - self.num_models_to_keep),
                            )
                        )
                except:
                    print("not enough models there yet, nothing to delete")

        else:
            # Save the predict model checkpoint
            torch.save(
                classification_model.state_dict(),
                os.path.join(self.opt.log_path, "classification_model_{}.ckpt".format(epoch)),
            )

            ### remove old model files to keep dir uncluttered
            try:
                os.remove(
                    os.path.join(
                        self.opt.log_path,
                        "classification_model_{}.ckpt".format(epoch - self.num_models_to_keep),
                    )
                )
            except:
                print("not enough models there yet, nothing to delete")

        if optimizer is not None:
            if isinstance(optimizer, list):
                for idx, optims in enumerate(optimizer):
                    torch.save(
                        optims.state_dict(),
                        os.path.join(
                            self.opt.log_path, "optim_{}_{}.ckpt".format(idx, epoch)
                        ),
                    )

                    try:
                        os.remove(
                            os.path.join(
                                self.opt.log_path,
                                "optim_{}_{}.ckpt".format(
                                    idx, epoch - self.num_models_to_keep
                                ),
                            )
                        )
                    except:
                        print("not enough models there yet, nothing to delete")
            else:
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        self.opt.log_path, "optim_{}.ckpt".format(epoch)
                    ),
                )
                try:
                    os.remove(
                        os.path.join(
                            self.opt.log_path,
                            "optim_{}.ckpt".format(
                                epoch - self.num_models_to_keep
                            ),
                        )
                    )
                except:
                    print("not enough models there yet, nothing to delete")
                


        # Save hyper-parameters
        with open(os.path.join(self.opt.log_path, "log.json"), 'w') as f:
            opt_dict = vars(self.opt).copy()
            cur_device = opt_dict.pop('device', None)
            json.dump(opt_dict, f, indent=2)
        with open(os.path.join(self.opt.log_path, "log.txt"), "w+") as cur_file:
            cur_file.write(str(self.opt))
            if accuracy is not None:
                cur_file.write("Top 1 -  accuracy: " + str(accuracy))
            if acc5 is not None:
                cur_file.write("Top 5 - Accuracy: " + str(acc5))
            if final_test and accuracy is not None:
                cur_file.write(" Very Final testing accuracy: " + str(accuracy))
            if final_test and acc5 is not None:
                cur_file.write(" Very Final testing top 5 - accuracy: " + str(acc5))

        # Save losses throughout training and plot
        np.save(
            os.path.join(self.opt.log_path, "train_loss"), np.array(self.train_loss)
        )
        
        for key, values in self.other_loss.items():
            np.save(
                os.path.join(self.opt.log_path, key), np.array(values)
            )
            print('saving loss {}: {}'.format(key, values))


        if self.val_loss is not None:
            #print('val loss {}'.format(self.val_loss))
            np.save(
                os.path.join(self.opt.log_path, "val_loss"), np.array(self.val_loss)
            )

        self.draw_loss_curve()

        if accuracy is not None:
            np.save(os.path.join(self.opt.log_path, "accuracy"), accuracy)

        if final_test:
            np.save(os.path.join(self.opt.log_path, "final_accuracy"), accuracy)
            np.save(os.path.join(self.opt.log_path, "final_loss"), final_loss)


    def draw_loss_curve(self):
        for idx, loss in enumerate(self.train_loss):
            lst_iter = np.arange(len(loss))
            plt.plot(lst_iter, np.array(loss), "-b", label="train loss")

            if self.loss_last_training is not None and len(self.loss_last_training) > idx:
                lst_iter = np.arange(len(self.loss_last_training[idx]))
                plt.plot(lst_iter, self.loss_last_training[idx], "-g")

            if self.val_loss is not None and len(self.val_loss) > idx:
                lst_iter = np.arange(len(self.val_loss[idx]))
                plt.plot(lst_iter, np.array(self.val_loss[idx]), "-r", label="val loss")

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc="upper right")
            # plt.axis([0, max(200,len(loss)+self.opt.start_epoch), 0, -round(np.log(1/(self.opt.negative_samples+1)),1)])

            # save image
            plt.savefig(os.path.join(self.opt.log_path, "loss_{}.png".format(idx)))
            plt.close()

    def append_train_loss(self, train_loss):
        for idx, elem in enumerate(train_loss):
            self.train_loss[idx].append(elem)

    def append_val_loss(self, val_loss):
        for idx, elem in enumerate(val_loss):
            self.val_loss[idx].append(elem)

    def append_other(self, other_loss, key):
        for idx, elem in enumerate(other_loss):
            self.other_loss[key][idx].append(elem)
