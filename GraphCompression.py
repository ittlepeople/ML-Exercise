from skimage import io
from sklearn.cluster import KMeans
import numpy as np

if __name__ == '__main__':
    image = io.imread('data/timg.jpg')
    io.imshow(image)
    io.show()

    rows = image.shape[0]
    cols = image.shape[1]
    # 把图变成一列，原来是 256*256
    image = image.reshape(image.shape[0] * image.shape[1], 3)
    kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
    kmeans.fit(image)

    clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
    labels = np.asarray(kmeans.labels_, dtype=np.uint8)
    # 重新组成这个图片
    labels = labels.reshape(rows, cols)

    print('簇的形状：', clusters.shape)
    np.save('codebook_test.npy', clusters)
    io.imsave('compressed_test.jpg', labels)
    image = io.imread('compressed_test.jpg')
    io.imshow(image)
    io.show()

