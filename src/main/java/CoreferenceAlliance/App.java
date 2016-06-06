package CoreferenceAlliance;


import java.io.File;

public class App
{
   public static void main( String[] args )
   {
       LSTM lstm = new LSTM();
       File f = new File( "DATA\\BIONLP\\Michael_XML\\PMID-1313226.xml" );
//       LSTM.AnnotatedDocument ad = lstm.new AnnotatedDocument(f);
      lstm.CreateNewNet();
      lstm.TrainFromXMLDir( "DATA\\BIONLP\\Michael_XML" );
   }
}