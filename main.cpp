#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include "rgb_euclidean_cluster_comparator.h"
#include <pcl/segmentation/organized_connected_component_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::ConstPtr CloudConstPtr;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer);

void
cloud_cb (const CloudConstPtr &cloud)
{
  // Show Cloud
  viewer->removeAllPointClouds ();
  viewer->addPointCloud (cloud);

  // Normal Estimation
  pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
  pcl::PointCloud<pcl::Normal>::Ptr normal_cloud (new pcl::PointCloud<pcl::Normal>);
  ne.setNormalEstimationMethod (ne.COVARIANCE_MATRIX);
  ne.setMaxDepthChangeFactor (0.02f);
  ne.setNormalSmoothingSize (20.0f);
  ne.setInputCloud (cloud);
  ne.compute (*normal_cloud);

  // Segment Planes
  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  std::vector<pcl::PlanarRegion<PointT>,
              Eigen::aligned_allocator<pcl::PlanarRegion<PointT> > > regions;
  std::vector<pcl::ModelCoefficients> model_coefficients;
  std::vector<pcl::PointIndices> inlier_indices;
  pcl::PointCloud<pcl::Label>::Ptr labels (new pcl::PointCloud<pcl::Label>);
  std::vector<pcl::PointIndices> label_indices;
  std::vector<pcl::PointIndices> boundary_indices;
  mps.setMinInliers (10000);
  mps.setAngularThreshold (pcl::deg2rad (3.0));
  mps.setDistanceThreshold (0.02);
  mps.setInputNormals (normal_cloud);
  mps.setInputCloud (cloud);
  mps.segmentAndRefine (regions, model_coefficients, inlier_indices,
                        labels, label_indices, boundary_indices);

  // Show Planar Regions
  char name[1024];
  unsigned char red[6] = {255, 0, 0, 255, 255, 0};
  unsigned char green[6] = {0, 255, 0, 255, 0, 255};
  unsigned char blue[6] = {0, 0, 255, 0, 255, 255};
  pcl::PointCloud<PointT>::Ptr contour (new pcl::PointCloud<PointT>);
  std::cout << "regions size: " << regions.size () << std::endl;
  for (size_t i = 0; i < regions.size (); ++i)
  {
    contour->points = regions[i].getContour ();
    sprintf (name, "plane_%02d", int (i));
    pcl::visualization::PointCloudColorHandlerCustom<PointT> color (
        contour, red[i % 6], green[i % 6], blue[i % 6]);
    if (!viewer->updatePointCloud (contour, color, name))
    {
      viewer->addPointCloud (contour, color, name);
    }
    viewer->setPointCloudRenderingProperties (
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
  }

  // Segment Objects
  if (regions.size () <= 0)
  {
    return;
  }
  std::vector<bool> plane_labels;
  plane_labels.resize (label_indices.size (), false);
  for (size_t i = 0; i < label_indices.size (); ++i)
  {
    if (label_indices[i].indices.size () > 10000)
    {
      plane_labels[i] = true;
    }
  }
  pcl::RGBEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>::Ptr comparator (
      new pcl::RGBEuclideanClusterComparator<PointT, pcl::Normal, pcl::Label>);
  comparator->setInputCloud (cloud);
  comparator->setLabels (labels);
  comparator->setExcludeLabels (plane_labels);
  comparator->setDistanceThreshold (0.01f, false);
  comparator->setColorThreshold (18.0f);
  pcl::PointCloud<pcl::Label> euclidean_labels;
  std::vector<pcl::PointIndices> euclidean_label_indices;
  pcl::OrganizedConnectedComponentSegmentation<PointT, pcl::Label> euclidean_segment (comparator);
  euclidean_segment.setInputCloud (cloud);
  euclidean_segment.segment (euclidean_labels, euclidean_label_indices);

  // Show Objects
  pcl::PointCloud<PointT>::Ptr cluster (new pcl::PointCloud<PointT>);
  int count = 0;
  for (size_t i = 0; i < euclidean_label_indices.size (); ++i)
  {
    if (euclidean_label_indices[i].indices.size () > 1000)
    {
      std::cout << "cluster " << count++ << std::endl;
      pcl::copyPointCloud (*cloud, euclidean_label_indices[i].indices, *cluster);
      sprintf (name, "cluster_%d", int (count));
      pcl::visualization::PointCloudColorHandlerCustom<PointT> color0 (
          cluster, red[count % 6], green[count % 6], blue[count % 6]);
      if (!viewer->updatePointCloud (cluster, color0, name))
      {
        viewer->addPointCloud (cluster, color0, name);
      }
      viewer->setPointCloudRenderingProperties (
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, name);
      viewer->setPointCloudRenderingProperties (
          pcl::visualization::PCL_VISUALIZER_OPACITY, 0.3, name);
    }
  }
}

int
main (int argc, char **argv)
{
    viewer->setCameraPosition (4.69428, 0.180855, -1.8609,
                               -1.54299, -1.6422, 5.1925,
                               0.133867, -0.981711, -0.13536);
    pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
    if (pcl::io::loadPCDFile (argv[1], *cloud) != 0)
    {
      std::cerr << "can't find pcd file" << std::endl;
      return -1;
    }
    cloud_cb (cloud);
    while (!viewer->wasStopped ())
    {
      viewer->spinOnce (100);
    }
    return 0;
}
