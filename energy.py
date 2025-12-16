################################## Algo - 1 (Works best) ##################################### 



# output = model(data)
# output_cap = model(data_cap)

# proj_feat = output["proj_features"]
# proj_feat_cap = output_cap["proj_features"]

# feat = output["features"]
# feat_cap = output["features"]

# esample = energy_model.langevin_sampling(feat, z_0 = feat_cap)
# esample_cap = energy_model.langevin_sampling(feat_cap, z_0 = feat)

# loss_con = lossfunction(proj_feat, proj_feat_cap) + F.mse_loss(esample.detach(), feat) + F.mse_loss(esample_cap.detach(), feat_cap)

# optimizer.zero_grad()
# loss_con.backward()
# optimizer.step()

# # training energy model
# pos_energy = energy_model(feat.detach(), feat_cap.detach())
# neg_energy = energy_model(esample.detach(), esample_cap.detach())
# energy_loss = pos_energy.mean() - neg_energy.mean()

# energy_optimizer.zero_grad()
# energy_loss.backward()
# energy_optimizer.step()


################################## Algo - 2 (Works well) ##################################### 

# output = model(data)
# output_cap = model(data_cap)

# proj_feat = output["proj_features"]
# proj_feat_cap = output_cap["proj_features"]

# feat = output["features"]
# # feat_cap = output["features"]
# feat_cap = output_cap["features"]

# # esample = energy_model.langevin_sampling(feat, z_0 = feat_cap)
# # esample_cap = energy_model.langevin_sampling(feat_cap, z_0 = feat)

# esample = energy_model.langevin_sampling(feat, z_0 = feat)
# esample_cap = energy_model.langevin_sampling(feat_cap, z_0 = feat_cap)

# loss_con = lossfunction(proj_feat, proj_feat_cap) + F.mse_loss(esample.detach(), feat) + F.mse_loss(esample_cap.detach(), feat_cap)

# optimizer.zero_grad()
# loss_con.backward()
# optimizer.step()

# # training energy model
# # pos_energy = energy_model(feat.detach(), feat_cap.detach())
# # neg_energy = energy_model(esample.detach(), esample_cap.detach())

# pos_energy = energy_model(feat.detach(), feat.detach()) + energy_model(feat_cap.detach(), feat_cap.detach())
# neg_energy = energy_model(esample.detach(), esample_cap.detach())
# energy_loss = pos_energy.mean() - neg_energy.mean()

# energy_optimizer.zero_grad()
# energy_loss.backward()
# energy_optimizer.step()