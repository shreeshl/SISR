im1 = im1/255.

layer_index = [3,8,15,22,29]
q = extract_hypercolumn(vgg,layer_index,Variable(torch.from_numpy(im1).view(3,250,250).unsqueeze(0).float()))

m = q.transpose(1,2,0).reshape(62500, -1)

cluster_labels = cluster.KMeans(n_clusters=2).fit_predict(m)

 