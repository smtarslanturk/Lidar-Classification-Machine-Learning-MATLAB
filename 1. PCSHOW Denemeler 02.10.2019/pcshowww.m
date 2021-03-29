numFaces = 600;
[x,y,z] = sphere(numFaces);
figure;
pcshow([x(:),y(:),z(:)]);
title('Sphere with Default Color Map');
xlabel('X');
ylabel('Y');
zlabel('Z');

I = im2double(imread('fiko.PNG'));
imshow(I);
title('Orginal fiko');

J = flipud(imresize(I,size(x)));
figure;
imshow(J);

figure;
pcshow([x(:),y(:),z(:)],reshape(J,[],3));
title('Sphere with Color Texture');
xlabel('X');
ylabel('Y');
zlabel('Z');


% I = ('JAX_066_PC3.txt');
% figure;
% pcshow([x(:),y(:),z(:)],reshape(I,[],3));
% title('Sphere with Color Texture');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');