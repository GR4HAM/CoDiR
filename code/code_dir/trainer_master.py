from cfg.config_general import cfg

def full_training(trainer, iterator_train, iterator_val, iterator_test):
    #responsible for organizing training loop and evaluating on best model after training
    best_score = -999

    trainer.load_networks(iterator_train.epoch_length, load_from=cfg.SAVE.SAVED_MODEL_LOC)
    trainer.prepare_for_training()
    for epoch in range(1, cfg.TRAIN.MAX_EPOCH+1):
        print(" -------------- epoch "+str(epoch)+" ------------------")

        trainer.train_epoch(epoch,iterator_train)
        trainer.calc_templates(iterator_train)

        if epoch > 0 and epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 :
            current_f1, current_precision, current_recall = trainer.evaluate_epoch(epoch, iterator_val, phase = "val")

            if current_f1 > best_score:
                best_score = current_f1

                current_results = ["best val score found in epoch", str(epoch),
                                   "f1 score", str(current_f1), "precision", str(current_precision),
                                   "recall", str(current_recall)]
                trainer.save_model(current_results)


    trainer.load_best_model()

    final_f1, final_precision, final_recall = trainer.evaluate_epoch(epoch, iterator_test, phase="test")

    print("final test f1: "+str(final_f1)+", P: "+str(final_precision)+", R: "+str(final_recall))

    trainer.close()

