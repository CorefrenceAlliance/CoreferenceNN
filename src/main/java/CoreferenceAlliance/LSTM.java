package CoreferenceAlliance;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.io.XMLEntityReader;
import gov.nih.nlm.ling.io.XMLImplicitRelationReader;
import gov.nih.nlm.ling.io.XMLReader;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.ImplicitRelation;
import gov.nih.nlm.ling.sem.SemanticItem;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.SerializationUtils;

import java.io.File;
import java.util.*;
import java.util.Random;


public class LSTM
{
   private MultiLayerNetwork net;

   private int do_output_index = 0;
   private int is_candidate_anaphor_index = 1;
   private int is_candidate_antecedent_index = 2;
   private int anaphor_match_index = 3;
   private int antecedent_match_index = 4;

   private int pos_index_offset = 5;
   private enum pos { CC, CD, DT, EX, FW, IN, JJ, JJR, JJS, LS, MD, NN, NNP,
         NNPS, NNS, PDT, POS, PRP, RB, RBR, RP, SYM, TO, UH, VB, VBD, VBN, VBP, VBZ,
         WDT, WP, WRB, LRB, RRB, VBG, RBS, PERIOD, COMMA, COLON, DACCENT, DSQUOTE }

   private int input_size = pos_index_offset + pos.values().length;
   private int first_layer_size = 100;
   private int second_layer_size = 100;
   private int output_size = 2;

   private int output_yes_index = 0;
   private int output_no_index = 1;

   Random random = new Random();


   public void CreateNewNet()
   {
      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .optimizationAlgo( OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT )
            .learningRate( 0.1 )
            .rmsDecay( 0.95 )
            .seed( 259742343 )
            .regularization( true )
            .l2( 0.001 )
            .weightInit( WeightInit.XAVIER )
            .updater( Updater.RMSPROP )
            .list( 3 )
            .layer( 0, new GravesLSTM.Builder()
                  .nIn( input_size )
                  .nOut( first_layer_size )
                  .activation( "tanh" )
                  .build() )
            .layer( 1, new GravesLSTM.Builder()
                  .nIn( first_layer_size )
                  .nOut( second_layer_size )
                  .activation( "tanh" )
                  .build() )
            .layer( 2, new RnnOutputLayer.Builder()
                  .nIn( second_layer_size )
                  .nOut( output_size )
                  .activation( "softmax" )
                  .build() )
            .backpropType( BackpropType.Standard )
            .pretrain( false )
            .backprop( true )
            .build();

      net = new MultiLayerNetwork( conf );
      net.init();
   }

   public void TrainFromXMLDir( String XML_Dir )
   {
      List< Document > docs = new ArrayList<>();

      int total_inputs = 0;
      int input_length = 0;

      for( File xml : new File( XML_Dir ).listFiles() )
      {
         Map< Class< ? extends SemanticItem >, List< String > > annotationTypes = new HashMap<>();
         annotationTypes.put( Entity.class, Arrays.asList( "Drug", "Drug_Class", "Substance", "SPAN",
               "DefiniteNP", "IndefiniteNP", "ZeroArticleNP", "DemonstrativeNP", "DistributiveNP",
               "PersonalPronoun", "PossessivePronoun", "DemonstrativePronoun", "DistributivePronoun",
               "ReciprocalPronoun", "RelativePronoun", "IndefinitePronoun" ) );
         annotationTypes.put( ImplicitRelation.class, Arrays.asList( "Anaphora", "Cataphora", "PredicateNominative", "Appositive" ) );
         XMLReader reader = new XMLReader();
         reader.addAnnotationReader( ImplicitRelation.class, new XMLImplicitRelationReader() );
         reader.addAnnotationReader( Entity.class, new XMLEntityReader() );
         Document doc = reader.load( xml.getAbsolutePath(), null, null, true, null, annotationTypes, null );

         docs.add( doc );
         input_length = Math.max( input_length, doc.getAllSurfaceElements().size() );

         LinkedHashSet< SemanticItem > relations = getAnaphoraRelations( doc );

         total_inputs += relations.size() * 2;     // Will generate as many negative cases as found positives
      }
      input_length++;   //final output step

//      System.out.println( total_inputs + " - " + input_length );

      INDArray features = Nd4j.zeros( total_inputs, input_size, input_length );
      INDArray labels = Nd4j.zeros( total_inputs, output_size, input_length );
      INDArray features_mask = Nd4j.zeros( total_inputs, input_length );
      INDArray labels_mask = Nd4j.zeros( total_inputs, input_length );

      int current_input = 0;

      for( int doc_num = 0; doc_num < docs.size(); doc_num++ )
      {
         Document doc = docs.get( doc_num );
         List< SurfaceElement > words = doc.getAllSurfaceElements();
         LinkedHashSet< SemanticItem > terms = getTerms( doc );
         LinkedHashSet< SemanticItem > relations = getAnaphoraRelations( doc );

         Map< SurfaceElement, Set< SurfaceElement > > used_relations = new HashMap<>();

         int relation_num = 0;
         // Positives
         for( SemanticItem relation : relations )
         {
            List< SurfaceElement > involved_words = doc.getSurfaceElementsInSpan( relation.getSpan() );
            SurfaceElement anaphor;
            SurfaceElement antecedent;
            if( SpanList.atLeft( involved_words.get( 0 ).getSpan(), involved_words.get( 1 ).getSpan() ) )
            {
               antecedent = involved_words.get( 0 );
               anaphor = involved_words.get( 1 );
            }
            else
            {
               anaphor = involved_words.get( 0 );
               antecedent = involved_words.get( 1 );
            }

            used_relations.putIfAbsent( antecedent, new HashSet<>() );
            used_relations.get( antecedent ).add( anaphor );

            // Set the output bit after the last word
//            features.put( new INDArrayIndex[] {
//                  NDArrayIndex.point( ( doc_num + 1 ) * relation_num ),
//                  NDArrayIndex.point( do_output_index ),
//                  NDArrayIndex.point( words.size() ) }, 1.0 );

//            System.out.println( current_input );
            features.putScalar( new int[] {
                  current_input,
                  do_output_index,
                  words.size() }, 1.0 );

            for( int word_num = 0; word_num < words.size(); word_num++ )
            {
               SurfaceElement word = words.get( word_num );
               boolean candidate_anaphor = anaphor.getSpan().equals( word.getSpan() );
               boolean candidate_antecedent = antecedent.getSpan().equals( word.getSpan() );
               boolean match_lemma_anaphor = anaphor.getLemma().equals( word.getLemma() );
               boolean match_lemma_antecdent = antecedent.getLemma().equals( word.getLemma() );

               String word_pos = cleanupPOS( word.getPos() );

               int pos_index;
               try
               {
                  pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();
               } catch( Exception e )
               {
                  System.out.println( word.getPos() );
                  System.out.println( "ADD THIS" );
                  throw e;
               }

               if( candidate_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_anaphor_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( candidate_antecedent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_antecedent_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( anaphor_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_antecdent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( antecedent_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               features.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( pos_index ),
                     NDArrayIndex.point( word_num ) }, 1.0 );

               features_mask.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( word_num ) }, 1.0 );
            }

            labels.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( output_yes_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );


            labels_mask.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            relation_num++;
            current_input++;
         }
         // Negatives
         for( ; relation_num < relations.size() * 2; relation_num++ )
         {
            SurfaceElement bad_anaphor;
            SurfaceElement bad_antecedent;
            do
            {
               int bad_anaphor_index = random.nextInt( terms.size() );
               int bad_antecedent_index = random.nextInt( terms.size() );

               Iterator< SemanticItem > iter = terms.iterator();
               for( int i = 0; i < bad_anaphor_index - 1; i++ )
                  iter.next();

               bad_anaphor = doc.getSurfaceElementsInSpan( iter.next().getSpan() ).get( 0 );

               iter = terms.iterator();
               for( int i = 0; i < bad_antecedent_index - 1; i++ )
                  iter.next();

               bad_antecedent = doc.getSurfaceElementsInSpan( iter.next().getSpan() ).get( 0 );
            }
            while(
                  used_relations.containsKey( bad_antecedent ) &&
                  used_relations.get( bad_antecedent ).contains( bad_anaphor ) );

            used_relations.putIfAbsent( bad_antecedent, new HashSet<>() );
            used_relations.get( bad_antecedent ).add( bad_anaphor );

            // Set the output bit after the last word
            features.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( do_output_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            for( int word_num = 0; word_num < words.size(); word_num++ )
            {
               SurfaceElement word = words.get( word_num );
               boolean candidate_anaphor = bad_anaphor.getSpan().equals( word.getSpan() );
               boolean candidate_antecedent = bad_antecedent.getSpan().equals( word.getSpan() );
               boolean match_lemma_anaphor = bad_anaphor.getLemma().equals( word.getLemma() );
               boolean match_lemma_antecdent = bad_antecedent.getLemma().equals( word.getLemma() );

               String word_pos = cleanupPOS( word.getPos() );

               int pos_index;
               try
               {
                  pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();
               } catch( Exception e )
               {
                  System.out.println( word.getPos() );
                  System.out.println( "ADD THIS" );
                  throw e;
               }

//               int pos_index = pos_index_offset + pos.valueOf( word_pos ).ordinal();

               if( candidate_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_anaphor_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( candidate_antecedent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( is_candidate_antecedent_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_anaphor )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( anaphor_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               if( match_lemma_antecdent )
                  features.put( new INDArrayIndex[] {
                        NDArrayIndex.point( current_input ),
                        NDArrayIndex.point( antecedent_match_index ),
                        NDArrayIndex.point( word_num ) }, 1.0 );

               features.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( pos_index ),
                     NDArrayIndex.point( word_num ) }, 1.0 );

               features_mask.put( new INDArrayIndex[] {
                     NDArrayIndex.point( current_input ),
                     NDArrayIndex.point( word_num ) }, 1.0 );
            }

            labels.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( output_no_index ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            labels_mask.put( new INDArrayIndex[] {
                  NDArrayIndex.point( current_input ),
                  NDArrayIndex.point( words.size() ) }, 1.0 );

            current_input++;
         }

      }


      DataSet ds = new DataSet( features, labels, features_mask, labels_mask );
      net.setListeners( new ScoreIterationListener( 1 ) );
      for( int epoch = 0; epoch < 100; epoch++ )
      {
         System.out.println( "Epoch: " + epoch );
         net.fit( ds );
         File out = new File( new Integer( epoch ).toString() );
         SerializationUtils.saveObject( net, out );
      }
   }

   private static String cleanupPOS( String pos )
   {
      pos = pos.replaceAll( "[-$]", "" );
      if( pos.equals( "." ) )
         pos = "PERIOD";
      if( pos.equals( "," ) )
         pos = "COMMA";
      if( pos.equals( ":" ) )
         pos = "COLON";
      if( pos.equals( "``" ) )
         pos = "DACCENT";
      if( pos.equals( "''" ) )
         pos = "DSQUOTE";
      return pos;
   }

   private static LinkedHashSet< SemanticItem > getAnaphoraRelations( Document doc )
   {
      LinkedHashSet< SemanticItem > s = (LinkedHashSet< SemanticItem >) doc.getAllSemanticItems().clone();
      s.removeIf( p -> !(p.getType().equals( "Anaphora" )) );
      return s;
   }

   private static LinkedHashSet< SemanticItem > getTerms( Document doc )
   {
      LinkedHashSet< SemanticItem > s = (LinkedHashSet< SemanticItem >) doc.getAllSemanticItems().clone();
      s.removeIf( p -> !(p.getId().charAt( 0 ) == 'T') );          // Doesn't seem to be a real way to get the terms
      return s;
   }

   public class AnnotatedDocument
   {
      private class Word_Info
      {
         String text;
         String lemma;
         String pos;
         Word_Info parent;
      }

      private class Mention
      {
         Word_Info info;
      }

      private class Relation
      {
         Mention anaphor;
         Mention antecedent;
      }

      List< Word_Info > words;
      List< Mention > mentions;    //contains references into words
      List< Relation > relations;  //contains references into mentions


      public AnnotatedDocument( File xml )
      {
         try
         {
            Map< Class< ? extends SemanticItem >, List< String > > annotationTypes = new HashMap<>();
            annotationTypes.put( Entity.class, Arrays.asList( "Drug", "Drug_Class", "Substance", "SPAN",
                  "DefiniteNP", "IndefiniteNP", "ZeroArticleNP", "DemonstrativeNP", "DistributiveNP",
                  "PersonalPronoun", "PossessivePronoun", "DemonstrativePronoun", "DistributivePronoun",
                  "ReciprocalPronoun", "RelativePronoun", "IndefinitePronoun" ) );
            annotationTypes.put( ImplicitRelation.class, Arrays.asList( "Anaphora", "Cataphora", "PredicateNominative", "Appositive" ) );
            XMLReader reader = new XMLReader();
            reader.addAnnotationReader( ImplicitRelation.class, new XMLImplicitRelationReader() );
            reader.addAnnotationReader( Entity.class, new XMLEntityReader() );
            gov.nih.nlm.ling.core.Document doc = reader.load( xml.getAbsolutePath(), null, null, true, null, annotationTypes, null );



//            // WORDS
//            List< SurfaceElement > surface_elements = doc.getAllSurfaceElements();
//            for( SurfaceElement se : surface_elements )
//            {
//               System.out.println( se.getLemma() + " : " + se.getPos()  + " : " + se.getSpan().toString() );
//            }
//
//
//            System.out.println();
//
//
////             RELATIONSHIPS
            LinkedHashSet< SemanticItem > semantic_items = doc.getAllSemanticItems();
            for( SemanticItem si : semantic_items )
            {
               System.out.println( si.getSpan().toString() );
               System.out.println( si.toShortString() );
               System.out.println( si.getType() );
               System.out.println( si.getId() );
               System.out.println( si.getAllSemtypes() );
//               System.out.println( doc.getSurfaceElementsInSpan( si.getSpan() ) );
//               for( SurfaceElement se : doc.getSurfaceElementsInSpan( si.getSpan() ) )
//               {
//                  System.out.println( se.getLemma() + " : " + se.getPos()  + " : " + se.getSpan().toString() );
//               }
//
//               for( SurfaceElement se : doc.getAllSurfaceElements() )
//               {
////                  if( SpanList.overlap( se.getSpan(), si.getSpan() ) )
////                     System.out.println( "OVERLAP: " + se.getLemma() + " : " + se.getSpan().toString() );
//               }

               System.out.println();
            }
//
//            System.out.println();
//
//            Tree doc_tree = doc.getDocumentTree();
//            System.out.println( doc_tree );



//            for( Class< ? extends SemanticItem > s : doc.getSemanticItems().keySet() )
//            {
//               System.out.println( "Annotation Type: " + s.getCanonicalName() + " | " + doc.getSemanticItems().get( s ).size() );
//               if( s.getCanonicalName().equals( "gov.nih.nlm.ling.sem.Entity" ) )
//               {
//                  //These are the terms
//                  //holds information on POS and location in doc
//                  for( SemanticItem si : doc.getSemanticItems().get( s ) )
//                  {
//                     System.out.println( "Annotation: " + si.toString() );
//                     String[] termElements = si.toString().split( "_" );
//                     for( String e : termElements )
//                     {
//                        System.out.println( e );
//                     }
//                     System.out.println();
//                  }
//                  System.out.println();
//               }
//               if( s.getCanonicalName().equals( "gov.nih.nlm.ling.sem.ImplicitRelation" ) )
//               {
//                  //These are the terms
//                  //holds information on POS and location in doc
//                  for( SemanticItem si : doc.getSemanticItems().get( s ) )
//                  {
//                     System.out.println( "Annotation: " + si.toString() );
//                     String[] relationElements = si.toString().split( "_" );
//                     System.out.println( "Relation Term: " + relationElements[ 5 ] );
//
////                            pos_tags.add(termElements[1]);
////                            lemmatized_tokens.add(termElements[3]);
//                  }
//               }
//            }
//                LinkedHashSet<SemanticItem> semanticItems = doc.getAllSemanticItems();
//                
//                for(SemanticItem semanticItem : semanticItems){
//                    Map<String, Object> features = semanticItem.getFeatures();
//                    
//                }

         } catch( Exception e )
         {
            e.printStackTrace();
         }

      }
   }

//   private class XMLAnnotationDirectoryIterator implements DataSetIterator
//   {
//      private class Token
//      {
//         private String text;
//         private String pos_tag;
//      }
//
//      List< File > files_remaining;
//      List< Token > tokens_current_file;
//      List< Integer > indices_mentions_remaining_current_file;
//      int batch_size;
//
//
//
//      public XMLAnnotationDirectoryIterator( String XML_Directory, int Batch_Size )
//      {
//         batch_size = Batch_Size;
//         for( File xml : new File( XML_Directory ).listFiles() )
//         {
//            files_remaining.add( xml );
//         }
//      }
//
//      public DataSet next()
//      {
//         return next(batch_size);
//      }
//
//      public DataSet next( int size )
//      {
//         int current_size = 0;
//
//         while( current_size < size && files_remaining.size() != 0 )
//         {
//
//         }
//
//         if( current_size != size ) throw NoSuchElementException;
//
//         DataSet result;
//         return result;
//      }
//   }
}
