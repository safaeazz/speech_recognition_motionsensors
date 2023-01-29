import tensorflow as tf
from functions import *
import numpy


# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -
# Step 1: Build the AE graph  
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -

# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -
# Step 2: Training (run the session)
# - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - -  --  -- -  - - -


def run_session(data,noise,noise_ratio,sparse,beta,lamda,num_input,num_hid,act_func,learning_rate,momentum,weight_init,
                     loss_func,opt,batch_size,epochs,
                     nameen,namedec,nb1,nb2):

  with tf.device('/gpu'):
  
    tf.reset_default_graph()
    X = tf.placeholder("float", [None, num_input])
    Xcorr = tf.placeholder("float", [None, num_input])
  

    w_en,w_dec = create_variables1(num_input,num_hid,weight_init,nameen,namedec)
    b_enc,b_dec = create_variables2(num_input,num_hid,nb1,nb2)  

    if noise == True:

      encode = encoder(Xcorr,act_func,w_en,b_enc)
    else:
      encode = encoder(X,act_func,w_en,b_enc)

    reconstruct = decoder(encode,act_func,w_dec,b_dec)



    if sparse == True:

      sparsity_target = lamda
      sparsity_weight = beta
      encode_mean = tf.reduce_mean (encode,axis=0)
      sparsity_loss = tf.reduce_sum (kl_divergence(sparsity_target,encode_mean))
      loss1 = cost_function(loss_func, X, reconstruct) 
      loss = loss1 + sparsity_weight * sparsity_loss
    
    else:

      loss = cost_function(loss_func, X, reconstruct) 

    
    tf.summary.scalar("mean_squared", loss)
    optimizer = optimize_alg(opt,learning_rate,loss,momentum)
    #saver = tf.train.Saver()
    init = tf.global_variables_initializer()

      
    with tf.Session() as sess:
           
          sess.run(init)    
          # Training cycle
          corruption_ratio = numpy.round(noise_ratio * num_input).astype(numpy.int)
          datacor = corrupt_input(data, corruption_ratio)
          for k in range(0,epochs): # for all epochs
        
              avg_cost = 0
              total_batch = int(len(data)/batch_size)
           
              # Loop over all batches        
              for j in range (0,data.shape[0], batch_size):            
                x_batch = data[j:j+batch_size]
                x_batchcorr = datacor[j:j+batch_size]
                # Run optimizer (backprop) and loss (to get loss value)
                #learning_rate = 0.01 if k < 500 else 0.001
                if noise == True:

                  l,  _ = sess.run([loss,optimizer], feed_dict={X: x_batch,
                                                              Xcorr: x_batchcorr})
                else:
                  
                  l,  _ = sess.run([loss,optimizer], feed_dict={X: x_batch})
                # Compute average loss
                avg_cost += l / total_batch

                
                #Display logs per epoch step
                #if k % 10 == 0:
                  #print("Epoch:", '%04d' % (k+1), "cost=", avg_cost)
                  
                # for validation:
                #l_val = sess.run([loss], feed_dict={X:validation_set})
                #print("Epoch:", k, "cost=",avg_cost," --- Validation cost at step",l_val)
                
          #print("cost over all epochs = ", avg_cost)
          #print("Optimization Finished!")

          #Save the variables to disk.
          #save_path = saver.save(sess, model_folder + "/" + model_name + ".ckpt")
          #print("Model saved in path: %s" % save_path)
          # get the features (code in the hidden layer)
          data = tf.constant(data, dtype=tf.float32)
  
          #get the model parameters -- generate features for classification
          #w, b = sess.run([w_en, b1]) 

          W1=w_en.eval()
          b1=b_enc.eval()  
          W1 = tf.constant(W1, dtype=tf.float32)
          b1 = tf.constant(b1, dtype=tf.float32)
          code = encoder(data,act_func,W1,b1)

          features= code.eval()
          return features,W1.eval(),b1.eval()









