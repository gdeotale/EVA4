import tensorflow as tf
from tqdm.notebook import tqdm
from /content/gdrive/My\ Drive/15anb/utils import utils
def train(model, criterion, device, train_loader, optimizer, epoch):
  model.train()
  loss1 = []
  pbar = tqdm(iter(train_loader), dynamic_ncols=True)
  for batch_idx, data in enumerate(pbar):
    if(batch_idx==0):
      t0 = time.time()
    #ta = time.time()
    data["f1_image"] = data["f1_image"].to(device)
    data["f2_image"] = data["f2_image"].to(device)
    data["f3_image"] = data["f3_image"].to(device)
    data["f4_image"] = data["f4_image"].to(device)
    #tb = time.time()
    optimizer.zero_grad()
    output, output_ = model(data)
    loss1 = criterion(output, data["f3_image"])
    loss2 = criterion(output_, data["f4_image"])
    loss = loss1 + 2*loss2
    loss.backward()
    optimizer.step()

    if(batch_idx != 0 and batch_idx%100==0):
      torch.cuda.empty_cache()
      iou3 = utils.calculate_iou(output.detach().cpu().numpy(), data["f3_image"].detach().cpu().numpy(),0.5)
      iou4 = utils.calculate_iou(output_.detach().cpu().numpy(), data["f4_image"].detach().cpu().numpy(),0.5)
      with train_summary_writer.as_default(): 
          tf.summary.scalar('iouf3', iou3, step=batch_idx)
          tf.summary.scalar('iouf4', iou4, step=batch_idx)
          tf.summary.scalar('lossf3', loss1.item(), step=batch_idx)
          tf.summary.scalar('lossf4', loss2.item(), step=batch_idx)
      t3 = time.time()
      print(t3-t0 )
      t0=t3
      print("Batch:" + str(batch_idx), " Epoch:"+str(epoch), " lOSSf3="+str(loss1.item()), " lOSSf4="+str(loss2.item()), 'iouf3', iou3, 'iouf4', iou4)
      for param_group in optimizer.param_groups:
        print("lr= ",param_group['lr'])   
      if(batch_idx%500==0):
        bools = True
        sample0 = data["f2_image"]
        sample = data["f3_image"]
        sample1 = data["f4_image"]
        utils.show(sample0, sample, sample1, output, output_, bools, str(epoch)+"_"+str(batch_idx))

def test(model, criterion, device, test_loader, epoch):
  model.eval()
  test_loss = 0
  correct = 0
  loss1 = []
  loss2 = []
  iou1 = []
  iou2 = []
  dice_loss1 = []
  dice_loss2 = []
  test_loss = []
  pbar = tqdm(iter(test_loader), dynamic_ncols=True)
  with torch.no_grad():
    for  batch_idx, data in enumerate(pbar):
       data["f1_image"] = data["f1_image"].to(device)
       data["f2_image"] = data["f2_image"].to(device)
       data["f3_image"] = data["f3_image"].to(device)
       data["f4_image"] = data["f4_image"].to(device)
       
       output, output_ = model(data)
       lossa = criterion(output, data["f3_image"])
       lossb = criterion(output_, data["f4_image"])
       ioua = utils.calculate_iou(output.detach().cpu().numpy(), data["f3_image"].detach().cpu().numpy(),0.5)
       ioub = utils.calculate_iou(output_.detach().cpu().numpy(), data["f4_image"].detach().cpu().numpy(),0.5)
       dice_lossa = dice_loss(output, data["f3_image"])
       dice_lossb = dice_loss(output_, data["f4_image"])
       loss = lossa + 2 * lossb
       loss1.append(lossa.item())
       loss2.append(lossb.item())
       test_loss.append(loss.item())
       iou1.append(ioua.item())
       iou2.append(ioub.item())
       test_loss.append(loss.item())
       dice_loss1.append(dice_lossa.item())
       dice_loss2.append(dice_lossb.item())
       pred = output.argmax(dim=1, keepdim=True)

       if(batch_idx%100==0):
          with test_summary_writer.as_default():
            tf.summary.scalar('test_loss', loss.item(), step=batch_idx)
            tf.summary.scalar('mask_loss', lossa.item(), step=batch_idx)
            tf.summary.scalar('depth_loss', lossb.item(), step=batch_idx)
            tf.summary.scalar('dice_loss_mask', dice_lossa.item(), step=batch_idx)
            tf.summary.scalar('dice_loss_depth', dice_lossb.item(), step=batch_idx)
            tf.summary.scalar('mask_iou', ioua.item(), step=batch_idx)
            tf.summary.scalar('depth_iou', ioub.item(), step=batch_idx)
          bools = False
          if(batch_idx%100==0):
            print("Batch:" + str(batch_idx), " Epoch:"+str(epoch), " lOSSf3="+str(lossa.item()), " lOSSf4="+str(lossb.item()), 'iouf3', ioua, 'iouf4', ioub)
          if(batch_idx%500==0):
            #break
            bools = True
            sample0 = data["f2_image"]
            sample = data["f3_image"]
            sample1 = data["f4_image"]
            utils.show(sample0, sample, sample1, output, output_, bools, "_test_"+str(epoch)+"_"+str(batch_idx))
  f = open("Output.txt","a")
  print("*********************************", file=f)
  print("Epoch "+str(epoch)+" Avg test loss: ",np.mean(test_loss)," Avg mask loss: ",np.mean(loss1), " Avg depth loss: ",np.mean(loss2), file=f)
  print("Epoch "+str(epoch)+" Avg mask iou: ",np.mean(iou1)," Avg depth iou: ",np.mean(iou2), file=f)
  print(" Avg mask dice loss: ",np.mean(dice_loss1), " Avg depth dice loss: ", np.mean(dice_loss2), file=f)
  f.close()