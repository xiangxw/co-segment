#ifndef PCL_SEGMENTATION_RGB_EUCLIDEAN_CLUSTER_COMPARATOR_H_
#define PCL_SEGMENTATION_RGB_EUCLIDEAN_CLUSTER_COMPARATOR_H_

#include <pcl/segmentation/boost.h>
#include <pcl/segmentation/euclidean_cluster_comparator.h>

namespace pcl
{
  /** \brief RGBEuclideanClusterComparator is a comparator used for find clusters with RGB and xyz. */
  template<typename PointT, typename PointNT, typename PointLT>
  class RGBEuclideanClusterComparator: public EuclideanClusterComparator<PointT, PointNT, PointLT>
  {
    public:
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloud PointCloud;
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudConstPtr PointCloudConstPtr;

      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudN PointCloudN;
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudNPtr PointCloudNPtr;
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudNConstPtr PointCloudNConstPtr;

      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudL PointCloudL;
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudLPtr PointCloudLPtr;
      typedef typename EuclideanClusterComparator<PointT, PointNT, PointLT>::PointCloudLConstPtr PointCloudLConstPtr;

      typedef boost::shared_ptr<RGBEuclideanClusterComparator<PointT, PointNT, PointLT> > Ptr;
      typedef boost::shared_ptr<const RGBEuclideanClusterComparator<PointT, PointNT, PointLT> > ConstPtr;

      using pcl::Comparator<PointT>::input_;

      /** \brief Empty constructor for RGBEuclideanClusterComparator. */
      RGBEuclideanClusterComparator ()
        : color_threshold_ (50.0f)
      {
      }

      /** \brief Destructor for RGBEuclideanClusterComparator. */
      virtual
      ~RGBEuclideanClusterComparator ()
      {
      }

      /** \brief Set the tolerance in color space between neighboring points, to be considered part of the same cluster.
        * \param[in] color_threshold The distance in color space
        */
      inline void
      setColorThreshold (float color_threshold)
      {
        color_threshold_ = color_threshold * color_threshold;
      }

      /** \brief Get the color threshold between neighboring points, to be considered part of the same plane. */
      inline float
      getColorThreshold () const
      {
        return (color_threshold_);
      }

      /** \brief Compare two neighboring points, by using normal information, euclidean distance, and color information.
        * \param[in] idx1 The index of the first point.
        * \param[in] idx2 The index of the second point.
        */
      bool
      compare (int idx1, int idx2) const
      {
        int dr = input_->points[idx1].r - input_->points[idx2].r;
        int dg = input_->points[idx1].g - input_->points[idx2].g;
        int db = input_->points[idx1].b - input_->points[idx2].b;
        //Note: This is not the best metric for color comparisons, we should probably use HSV space.
        float color_dist = static_cast<float> (dr*dr + dg*dg + db*db);
        return ( (color_dist < color_threshold_)
                 && EuclideanClusterComparator<PointT, PointNT, PointLT>::compare (idx1, idx2));
      }

    protected:
      float color_threshold_;
  };
}

#endif // PCL_SEGMENTATION_RGB_EUCLIDEAN_CLUSTER_COMPARATOR_H_
