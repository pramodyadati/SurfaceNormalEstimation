function visualize_whitened_depthmaps(dm,img)

[Xwh,avg,invMat, whMat] = whiten(dm(:));
Xwh=reshape(Xwh,size(dm));
figure;
imshow(img);
figure;
imagesc(imadjust(mat2gray(Xwh)));

[a,b]=depthToCloud(Xwh);
[x,y,z]=surfnorm(a(:,:,1),a(:,:,2),a(:,:,3));
figure;
imagesc(cat(3,imadjust(mat2gray(x)),imadjust(mat2gray(y)),imadjust(mat2gray(z))));