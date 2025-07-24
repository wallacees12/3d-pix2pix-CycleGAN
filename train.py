import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer

# Try to import tqdm, fallback to simple range if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available, using simple progress")
    def tqdm(iterable, desc="", total=None, leave=True):
        """Simple fallback for tqdm"""
        return iterable

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

# Main training loop with tqdm progress bars
total_epochs = opt.niter + opt.niter_decay
epoch_pbar = tqdm(range(opt.epoch_count, total_epochs + 1), desc="Training", unit="epoch") if TQDM_AVAILABLE else range(opt.epoch_count, total_epochs + 1)

for epoch in epoch_pbar:
    epoch_start_time = time.time()
    epoch_iter = 0
    
    # Progress bar for iterations within epoch
    if TQDM_AVAILABLE:
        iter_pbar = tqdm(enumerate(dataset), total=dataset_size, desc=f"Epoch {epoch}", leave=False, unit="batch")
    else:
        iter_pbar = enumerate(dataset)
        print(f"Epoch {epoch}/{total_epochs}")
    
    for i, data in iter_pbar:
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            
            # Update progress bars with loss information
            if TQDM_AVAILABLE:
                loss_str = " | ".join([f"{k}: {v:.3f}" for k, v in errors.items()])
                iter_pbar.set_postfix_str(loss_str)
                
                # Update epoch progress bar less frequently to avoid clutter
                if i % max(1, dataset_size // 10) == 0:  # Update every 10% of epoch
                    epoch_pbar.set_postfix_str(f"Loss: {list(errors.values())[0]:.3f}")
            else:
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            if TQDM_AVAILABLE:
                tqdm.write(f'💾 Saving latest model (epoch {epoch}, total_steps {total_steps})')
            else:
                print(f'💾 Saving latest model (epoch {epoch}, total_steps {total_steps})')
            model.save('latest')

    # End of epoch
    epoch_time = time.time() - epoch_start_time
    
    if epoch % opt.save_epoch_freq == 0:
        if TQDM_AVAILABLE:
            tqdm.write(f'💾 Saving model at end of epoch {epoch}, iters {total_steps}')
        else:
            print(f'💾 Saving model at end of epoch {epoch}, iters {total_steps}')
        model.save('latest')
        model.save(epoch)

    # Update epoch progress bar
    if TQDM_AVAILABLE:
        epoch_pbar.set_description(f"Training - Epoch {epoch}/{total_epochs} ({epoch_time:.0f}s)")
    else:
        print(f'✅ Epoch {epoch}/{total_epochs} completed in {epoch_time:.0f}s')

    if epoch > opt.niter:
        model.update_learning_rate()

# Training completed
if TQDM_AVAILABLE:
    tqdm.write("🎉 Training completed!")
else:
    print("🎉 Training completed!")
