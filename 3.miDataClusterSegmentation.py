import argparse
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh


# 1. Adquisicion de datos

def adquisicion_de_datos(dataset, clase_objeto, objeto, n_captura, n_frame):

    matrix_xyz = np.loadtxt(f"./{dataset}/xyzConf/{clase_objeto}/{objeto}_C{n_captura}_F{n_frame}.txt")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(matrix_xyz)
    print("Nube original: ")
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
    o3d.visualization.draw_geometries([pcd])

    return pcd


# 2. Calcular plano principal (pared o fondo) con RANSAC, segmentarlo, obtener la nube de outliers de ese plano

def segmentacion(pcd, umbral, clase_objeto, objeto, n_captura, n_frame, results):

    plane_model, inliners = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Eliminado el plano : {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inliner_cloud = pcd.select_by_index(inliners)
    inliner_cloud.paint_uniform_color([1, 0, 0])
    outlier_cloud = pcd.select_by_index(inliners, invert=True)

    print("Nube con plano segmentado a eliminar resaltado y resto de la nube: ")
    o3d.visualization.draw_geometries([inliner_cloud, outlier_cloud])

    print("Resto de la nube tras eliminar el plano: ")
    o3d.visualization.draw_geometries([outlier_cloud])  # puntos que no pertenecen

    # Eliminación de los puntos que quedan sueltos antes de clusterizar para mejorar el clusterizado y obtener el menor
    # numero de puntos no deseados

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Ruido a eliminar (rojo) y nube resultante (gris): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # elimina puntos que tienen menos del numero de vecinos indicados en la esfera "creada"
    cl, ind = outlier_cloud.remove_radius_outlier(nb_points=16, radius=0.05)

    display_inlier_outlier(outlier_cloud, ind)
    print("Nube resultante tras filtrar con el radio:")
    o3d.visualization.draw_geometries([cl])  # nube ya limpia

    # 3. Segmentacion (clusterizacion de la nube) de los puntos que no pertenecen al plano con DBSCAN()

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(cl.cluster_dbscan(eps=0.15, min_points=10, print_progress=True))
        max_label = labels.max() + 1
        print(f"La nube de puntos tiene: {max_label} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        cl.colors = o3d.utility.Vector3dVector(colors[:, :3])
        print("Nube clusterizada: ")
        o3d.visualization.draw_geometries([cl])

        clusters = []
        boxes = []

        # 3.2 Muestra Todos los clusters con su respectiva distancia, los que superan el umbral son posibles objetos y se guardan en el Dataset
        for i in range(max_label):
            id = np.where(labels == i)[0]
            pcd_i = cl.select_by_index(id)
            print(f"Cluster {i}: {pcd_i}")
            # resalto los cluster con una caja alineada roja
            aabb = pcd_i.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            clusters.append(pcd_i)
            boxes.append(aabb)
            if len(pcd_i.points) >= umbral:
                # muestra el objeto
                o3d.visualization.draw_geometries([pcd_i])
                # guarda la nube en el dataset como .ply y .txt
                pcd_i_xyzarray = np.asarray(pcd_i.points)
                np.savetxt(f"{results}txt/{clase_objeto}/{objeto}_C{n_captura}_F{n_frame}_Cluster{i}.txt", pcd_i_xyzarray)  # guarda la nube del cluster como matrizxyz.txt
                o3d.io.write_point_cloud(f"{results}ply/{clase_objeto}/{objeto}_C{n_captura}_F{n_frame}_Cluster{i}.ply", pcd_i)  # guarda la nube del cluster como .ply


def main():
    # valores por defecto
    umbral = 2048
    directorio = "MiDataCluster"
    directorio_results = "MiDataCluster/clustersConf/"
    clase_objeto = "silla"
    objeto = "SillaA"
    n_captura = 11
    n_frame = 0

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("-d", "--dataset", type=str, default=directorio, help="ruta al directorio principal donde se encuentra el dataset a segmentar")
    parser.add_argument("-r", "--results", type=str, default=directorio_results, help="ruta al directorio donde se guarda los clusters segmentados")
    parser.add_argument("-c", "--clase_objeto", type=str, default=clase_objeto, help="clase del objeto a segmentar")
    parser.add_argument("-o", "--objeto", type=str, default=objeto, help="nombre del objeto a segmentar")
    parser.add_argument("-n", "--n_captura", type=int, default=n_captura, help="numero de captura a segmentar")
    parser.add_argument("-f", "--n_frame", type=int, default=n_frame, help="numero de captura a segmentar")
    parser.add_argument("-p", "--num_points", type=int, default=umbral, help="numero de puntos a muestrear la nube")
    options = parser.parse_args()

    pcd = adquisicion_de_datos(options.dataset, options.clase_objeto, options.objeto, options.n_captura, options.n_frame)
    segmentacion(pcd, options.num_points, options.clase_objeto, options.objeto, options.n_captura, options.n_frame, options.results)


if __name__ == "__main__":
    main()