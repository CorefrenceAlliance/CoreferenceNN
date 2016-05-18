package gov.nih.nlm.ling.sem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.Section;
import gov.nih.nlm.ling.core.Sentence;
import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.core.SynDependency;
import gov.nih.nlm.ling.transform.CoordinationTransformation;
import gov.nih.nlm.ling.transform.DependencyTransformation;
import gov.nih.nlm.ling.util.CollectionUtils;
import gov.nih.nlm.ling.util.DiscourseUtils;
import gov.nih.nlm.ling.util.SemUtils;

/**
 * This class contains static utility methods to find and interpret coordination in text. 
 * 
 * @author Halil Kilicoglu
 *
 */
public class ConjunctionDetection {
	private static Logger log = Logger.getLogger(ConjunctionDetection.class.getName());
	
	private static final List<String> CONJ_SEPARATORS = Arrays.asList(",",";");
	private static final List<String> AND_COORDINATORS = Arrays.asList("and","to","also","well","&",",","plus");
	private static final List<String> OR_COORDINATORS = Arrays.asList("or",",");
	// rather than?
	private static final List<String> NEG_COORDINATORS = Arrays.asList("not","instead","rather");
	
	/**
	 * Annotates all conjunction relations in the document.
	 * 
	 * @param doc	the document to annotate
	 * 
	 * @deprecated
	 */
	public static void identifyConjunctions(Document doc) {
		List<Sentence> sentences = doc.getSentences();
		for (Sentence sent: sentences) {
			identifyConjunctions(sent);
		}
	}
	
	/**
	 * Annotates all conjunctions relations in a sentence.<p>
	 * Uses the initial dependency relations.
	 * 
	 * @param sent	the sentence to annotate
	 * 
	 * @deprecated	{@code #identifyConjunctionRelationsFromTransformation(Sentence)}
	 */
	public static synchronized void identifyConjunctions(Sentence sent) {
		Document doc = sent.getDocument();
		SemanticItemFactory sif = doc.getSemanticItemFactory();
		List<SynDependency> embeddings = sent.getEmbeddings();
		for (SynDependency e: embeddings) {
			if (e.getType().startsWith("conj_")) {
				SurfaceElement gov = e.getGovernor();
				SurfaceElement dep = e.getDependent();
				Set<SemanticItem> govSem = gov.getSemantics();
				Set<SemanticItem> depSem = dep.getSemantics();
				if (govSem == null || depSem == null) continue;
				SurfaceElement conjToken = conjSurfaceElement(sent, e);
				// TODO: Does not allow implicitly indicated coordinations
				if (conjToken == null) continue;
				for (SemanticItem gSem : govSem) {
					Set<SemanticItem> gc = getConjunctSemanticItems(doc, gSem);
					for (SemanticItem dSem: depSem) {
						if (gc!=null && gc.contains(dSem)) continue;
						String type = semanticallyConsonant(dSem, gSem);
						log.log(Level.FINEST,"Conjunction type: {0}.", new Object[]{type});
						if (type != null) {
							if (gc == null) {
								LinkedHashSet<SemanticItem> cc = new LinkedHashSet<SemanticItem>();
								cc.add(gSem); cc.add(dSem);
								Predicate pred = sif.newPredicate(doc, conjToken.getSpan(),conjToken.getHead().getSpan(),type);
								sif.newConjunction(doc, type, pred, cc);
							}
							else {
								Conjunction pred = getConjunctionsWithSemanticItem(doc, gSem);
								pred.addSemanticItem(dSem);
								log.log(Level.FINEST,"Added term {0} as a conjunct to {1}.", new Object[]{dSem.toShortString(),pred.toShortString()});
							}	
						}
					}
				}
			}
		}
	}
	
	/**
	 * Annotates all conjunction relations in a sentence using its embedding structure.<p>
	 * In addition to embeddings, it checks for semantic consonance of the potential conjuncts.
	 * It relies on embeddings generated by {@link CoordinationTransformation}, so the results
	 * may not be accurate if the sentence has not been transformed by this transformation.
	 * 
	 * @param sent	the sentence to annotate
	 */
	public static synchronized void identifyConjunctionRelationsFromTransformation(Sentence sent) {
		if (sent == null || sent.getSurfaceElements() == null) return;
		List<Class<? extends DependencyTransformation>> applied = sent.getTransformations();
		if (applied.contains(CoordinationTransformation.class) == false) {
			log.log(Level.WARNING,"CoordinationTransformation is a prerequisite and has not been performed, the results may not be accurate.");
		}
		Document doc = sent.getDocument();
		SemanticItemFactory sif = doc.getSemanticItemFactory();
		List<SynDependency> embeddings = sent.getEmbeddings();
		List<Conjunction> conjs = new ArrayList<>();
		for (SurfaceElement su: sent.getSurfaceElements()) {
			List<SynDependency> ccs = SynDependency.outDependenciesWithType(su, embeddings, "cc", true);
//			int num = 1;
			if (ccs.size() > 1) {
				List<Collection<SemanticItem>> semLists = new ArrayList<Collection<SemanticItem>>();
				for (SynDependency cc: ccs) {
					SurfaceElement dep = cc.getDependent();
					LinkedHashSet<SemanticItem> depSem = dep.getSemantics();
					if (depSem == null || depSem.size() == 0) continue;
					semLists.add(dep.getSemantics());
//					num *= dep.getSemantics().size();
				}
				if (semLists.size() < 2) continue;
				Collection<List<SemanticItem>> perms = CollectionUtils.generatePermutations(semLists);
				List<SemanticItem> ccSems = null;
				String type = null;
				for (List<SemanticItem> perm : perms) {
					SemanticItem first = perm.get(0);
					for (int i=1; i < perm.size(); i++) {
						type = semanticallyConsonant(first, perm.get(i));
					}
					if (type != null) {
						if (ccSems == null) ccSems = perm;
						else ccSems.addAll(perm);
					}
				}
				if (ccSems == null || type == null) continue;
				Predicate pred = sif.newPredicate(doc, su.getSpan(),su.getHead().getSpan(),type);
				Conjunction rel = sif.newConjunction(doc, type, pred, new LinkedHashSet<SemanticItem>(ccSems));
				conjs.add(rel);
			}
		}
		consolidateConjunctions(conjs);
	}
	
	// If multiple conjunction relations only separated by punctuations or coordinators have been generated,
	// this can consolidate them into a single conjunction, provided that the elements are semantically consonant.
	// TODO may consider removal, as it had little positive influence on SPL training set (0.4839 vs. 0.4871 for anaphora)
	private static synchronized void consolidateConjunctions(List<Conjunction> conjs) {
		if (conjs == null || conjs.size() ==0) return;
		List<List<Conjunction>> consolidated = new ArrayList<List<Conjunction>>();
		Document doc = conjs.get(0).getDocument();
		List<Conjunction> added = new ArrayList<>();
		for (int i=0; i < conjs.size(); i++) {
			Conjunction fconj = conjs.get(i);
			Span fsp = fconj.getSpan().asSingleSpan();
			SemanticItem farg = fconj.getArgItems().get(0);
			for (int j=i+1;j <conjs.size(); j++) {
				Conjunction sconj = conjs.get(j);
				if (added.contains(sconj)) continue;
				SemanticItem sarg = sconj.getArgItems().get(0);
				Span ssp = sconj.getSpan().asSingleSpan();
				Span left = fsp; Span right = ssp;
				if (Span.atLeft(ssp,fsp)) { left = ssp; right = fsp;}
				if ((Span.overlap(left,right) || spanWithConjElementsOnly(doc,new Span(left.getEnd()+1,right.getBegin())))
					 && semanticallyConsonant(farg,sarg) != null) {
					boolean exists = false;
					for (List<Conjunction> cons: consolidated) {
						if (cons.contains(fconj)) {
							exists = true;
							cons.add(sconj);
							break;
						}
					}
					if (!exists) {
						List<Conjunction> nc = new ArrayList<Conjunction>();
						nc.add(fconj); nc.add(sconj);
						consolidated.add(nc);
					}
					added.add(sconj);
					if (added.contains(fconj) == false) added.add(fconj);
				}		
			}
		}
		for (List<Conjunction> lc: consolidated) {
			LinkedHashSet<SemanticItem> allArgItems = new LinkedHashSet<SemanticItem>();
			Predicate pr = null;
			String type = "";
			List<Conjunction> remove = new ArrayList<Conjunction>();
			for (Conjunction cc: lc) {
				allArgItems.addAll(cc.getArgItems());
				pr = cc.getPredicate();
				type = cc.getType();
				remove.add(cc);
			}
			Predicate pr1 = doc.getSemanticItemFactory().newPredicate(doc, pr.getSpan(),pr.getHeadSpan(),pr.getText()); 
			Conjunction nc = doc.getSemanticItemFactory().newConjunction(doc, type, pr1, allArgItems);
			for (Conjunction rm: remove) {
				doc.removeSemanticItem(rm);
				log.log(Level.FINEST,"Consolidating existing conjunction relation {0} into new conjunction relation {1}.", new Object[]{rm.toShortString(),nc.toShortString()});
			}
		}
	}
	
	/**
	 * Annotates inter-sentential, discourse level conjunction relations involving entities.<p>
	 * It uses sections of the document and some basic heuristics 
	 * to identify possibly conjoined items. 
	 * 
	 * @param doc	the document to annotate
	 */
	// TODO Specific to SPL data, needs to be expanded to be generally useful
	public static synchronized void identifyDiscourseLevelConjunctions(Document doc) {
		List<Section> sections = doc.getSections();
		if (sections == null) return;
		// for now, only entities
		for (Section sect: sections) {
			List<Section> subsects = sect.getSubSections();
			if (subsects.size() == 0) 
				identifyDiscourseLevelConjunctionsInSection(doc,sect);
			else {
				for (Section subsect: subsects) {
					identifyDiscourseLevelConjunctionsInSection(doc,subsect);
				}
			}
		}
	}
	
/*	public static synchronized void identifyDiscourseLevelConjunctionsInSection(Document doc, Section section) {
		SemanticItemFactory sif = doc.getSemanticItemFactory();
		Map<String,List<Entity>> entityGroups = new HashMap<String,List<Entity>>();
		Span sectSpan = section.getSpan();
		LinkedHashSet<SemanticItem> entities = Document.getSemanticItemsByClassSpan(doc, Entity.class, new SpanList(sectSpan), false);
		List<Sentence> sectSentences = section.getSentences();
		for (SemanticItem sem: entities) {
			Entity ent = (Entity)sem;
//			if (getConjunctionRelation(doc,ent) != null) continue;
			LinkedHashSet<String> semgroups = SemUtils.getSemGroups(ent);
			for (String sg: semgroups) {
				List<Entity> sgents = new ArrayList<Entity>();
				if (entityGroups.containsKey(sg)) {
					sgents = entityGroups.get(sg);
					if (semanticallyConsonant(sgents.get(0),ent) == null) continue;
				}
				sgents.add(ent);
				entityGroups.put(sg, sgents);
			}
		}
		for (String g: entityGroups.keySet()) {
			List<Entity> ents = entityGroups.get(g);
//			LinkedHashSet<SemanticItem> potentialConjoined = new LinkedHashSet<SemanticItem>();
			Map<SemanticItem,String> enumeratedConjuncts =new HashMap<SemanticItem,String>();
			for (Entity ent: ents) {
				String type=enumeratedEntityType(ent);
				if (type != null) 
					enumeratedConjuncts.put(ent, type);
//				if (enumeratedEntity(ent))
//					potentialConjoined.add(ent);
			}
			if (enumeratedConjuncts.size() > 1) {
				// group them by sentence position
				Map<Integer,LinkedHashSet<SemanticItem>> sentencePosGroups = new HashMap<Integer,LinkedHashSet<SemanticItem>>();
				LinkedHashSet<SemanticItem> current = new LinkedHashSet<SemanticItem>(); 
				String currentType = "";
				int sentId = -1;
				for (Sentence sent: sectSentences) {
					LinkedHashSet<SemanticItem> sentEntities = new LinkedHashSet<SemanticItem>(
							Document.getSemanticItemsByClassSpan(doc, Entity.class, new SpanList(sent.getSpan()), false));
					sentEntities.retainAll(enumeratedConjuncts.keySet());
					if (sentEntities.size() > 0) {
						String temp = enumeratedConjuncts.get(sentEntities.iterator().next());
						if (current.size() == 0 || currentType.equals(temp) == false) {
							sentId = Integer.parseInt(sent.getId().substring(1));
							sentencePosGroups.put(sentId, sentEntities);
						} else {
							sentencePosGroups.get(sentId).addAll(sentEntities); 	
						}
						current.addAll(sentEntities);
						currentType = temp;
					} else {
						current = new LinkedHashSet<SemanticItem>();
					}
				}
//				Map<Integer,Integer> sentenceMappings = new HashMap<Integer,Integer>();
				List<SemanticItem> remove = new ArrayList<SemanticItem>();
				for (SemanticItem ent: potentialConjoined) {

					if (sentenceMappings.containsKey(sentId)) {
						sentencePosGroups.get(sentenceMappings.get(sentId)).add(ent);
					} else if (sentenceMappings.containsKey(sentId+1)) {
						sentencePosGroups.get(sentenceMappings.get(sentId+1)).add(ent);
						sentenceMappings.put(sentId, sentId+1);
					} else if (sentenceMappings.containsKey(sentId-1)) {
						sentencePosGroups.get(sentenceMappings.get(sentId-1)).add(ent);
						sentenceMappings.put(sentId, sentId-1);
					} else {
						LinkedHashSet<SemanticItem> ns = new LinkedHashSet<SemanticItem>();
						ns.add(ent); sentencePosGroups.put(sentId, ns);
						sentenceMappings.put(sentId, sentId);
					}
				}
//				for (SemanticItem ent : potentialConjoined) {
				for (Integer gr: sentencePosGroups.keySet()) {
					LinkedHashSet<SemanticItem> existingConjoined= new LinkedHashSet<SemanticItem>();
					LinkedHashSet<SemanticItem> grItems = sentencePosGroups.get(gr);
					// singleton
					if (grItems.size() == 1) continue;
					for (SemanticItem ent: grItems) {
						Set<SemanticItem> conjoined = getConjoinedItems(doc,ent);
						// ADD conjoined items to ent as well
						if (conjoined != null) { 
							existingConjoined.addAll(conjoined);
							Conjunction existingConj = getConjunctionRelation(doc,ent);
							remove.add(existingConj);
						}
					}
					grItems.addAll(existingConjoined);
					Entity first = (Entity)grItems.iterator().next();
					Predicate pr = sif.newPredicate(doc,first.getSpan(),first.getHeadSpan(), g);
					Conjunction discRel = doc.getSemanticItemFactory().newConjunction(doc, g, pr, grItems);
					log.debug("DISCOURSE LEVEL CONJUNCTION: " + discRel.toString());
					for (SemanticItem rm: remove) {
						log.debug("REPLACED, SO REMOVING: " + rm.toString());
						doc.removeSemanticItem(rm);
					}
				}
			}
		}
	}*/
	

	/**
	 * Annotates discourse-level conjunction relations in one section of the document.<p>
	 * Finds conjunction relations involving entities only at this time. It makes use of 
	 * semantic group/type information and list/item structure.
	 * 
	 * @param doc		the document
	 * @param section	the section to annotate
	 */
	// TODO Find a way perhaps of making sure semantic objects created here do not conflict with regular Conjunction objects.
	// TODO Use ImplicitRelation
	public static synchronized void identifyDiscourseLevelConjunctionsInSection(Document doc, Section section) {
		SemanticItemFactory sif = doc.getSemanticItemFactory();
		Map<String,List<Entity>> entityGroups = new HashMap<String,List<Entity>>();
		Span sectSpan = section.getTextSpan();
		LinkedHashSet<SemanticItem> entities = Document.getSemanticItemsByClassSpan(doc, Entity.class, new SpanList(sectSpan), false);
		// group entities in the section by semantic group
		for (SemanticItem sem: entities) {
			Entity ent = (Entity)sem;
			LinkedHashSet<String> semgroups = SemUtils.getSemGroups(ent);
			for (String sg: semgroups) {
				List<Entity> sgents = new ArrayList<>();
				if (entityGroups.containsKey(sg)) {
					sgents = entityGroups.get(sg);
					if (semanticallyConsonant(sgents.get(0),ent) == null) continue;
				}
				sgents.add(ent);
				entityGroups.put(sg, sgents);
			}
		}
		for (String g: entityGroups.keySet()) {
			List<Entity> ents = entityGroups.get(g);
			LinkedHashSet<SemanticItem> potentialConjoined = new LinkedHashSet<SemanticItem>();
			// check whether the entities are part of a list
			for (Entity ent: ents) {
				String type=enumeratedEntityType(ent);
				if (type != null) 
					potentialConjoined.add(ent);
			}
			if (potentialConjoined.size() > 1) {
				List<SemanticItem> remove = new ArrayList<>();
				LinkedHashSet<SemanticItem> existingConjoined= new LinkedHashSet<>();
				for (SemanticItem ent : potentialConjoined) {
					Set<SemanticItem> conjoined = getConjunctSemanticItems(doc,ent);
					// ADD conjoined items to ent as well
					if (conjoined != null) { 
						existingConjoined.addAll(conjoined);
						Conjunction existingConj = getConjunctionsWithSemanticItem(doc,ent);
						remove.add(existingConj);
					}
				}
				potentialConjoined.addAll(existingConjoined);
				Entity first = (Entity)potentialConjoined.iterator().next();
				Predicate pr = sif.newPredicate(doc,first.getSpan(),first.getHeadSpan(), g);
				Conjunction discRel = doc.getSemanticItemFactory().newConjunction(doc, g, pr, potentialConjoined);
				log.log(Level.FINE,"Discourse-level conjunction generated: {0}.", new Object[]{discRel.toShortString()});
				for (SemanticItem rm: remove) {
					log.log(Level.FINER,"Replaced {0} with {1}.", new Object[]{rm.toShortString(),discRel.toShortString()});
					doc.removeSemanticItem(rm);
				}
			}
		}
	}
	
	// Very rudimentary way of determining whether the entity is an item in a list and the type of the list.
	private static String enumeratedEntityType(Entity ent) {
		SurfaceElement surf = ent.getSurfaceElement();
		Sentence sent = surf.getSentence();
		Span prev = new Span(sent.getSpan().getBegin(),ent.getSpan().getBegin());
		String enumerationType = "";
		if (prev.length() > 0) {
			List<SurfaceElement> prevs = sent.getDocument().getSurfaceElementsInSpan(prev);
			for (SurfaceElement se : prevs) {
				if (se.containsLemma("-")) enumerationType += "-";
				else if (se.getCategory().equals("CD")) enumerationType += "NUM";
				else return null;
			}
		}  
		return enumerationType;
	}
	
	/**
	 * Finds <code>Conjunction</code> objects indicated by the textual unit <var>coord</var> (the coordinator).
	 * 
	 * @param coord	a textual unit
	 * @return		the set of <code>Conjunction</code> objects, or empty set if none
	 */
	public static LinkedHashSet<SemanticItem> filterByConjunctions(SurfaceElement coord) {
		LinkedHashSet<SemanticItem> conjs = new LinkedHashSet<>();
		LinkedHashSet<SemanticItem> rels = coord.filterByRelations();
		if (rels == null || rels.size() ==0) return conjs;
		for (SemanticItem rel: rels) {
			if (rel instanceof Conjunction) conjs.add(rel); 
		}
		return conjs;
	}
	
	/**
	 * Finds all semantic items that are coordinated by the textual unit <var>coord</var>.
	 * 
	 * @param coord	a textual unit
	 * @return		the set of semantic items coordinated by the textual unit, or empty set if none
	 */
	public static LinkedHashSet<SemanticItem> getConjunctionArguments(SurfaceElement coord) {
		LinkedHashSet<SemanticItem> conjs = filterByConjunctions(coord);
		LinkedHashSet<SemanticItem> args = new LinkedHashSet<>();
		if (conjs.size() == 0) return args;
		for (SemanticItem conj: conjs) {
			if (conj instanceof Conjunction) {
				Conjunction con = (Conjunction)conj;
				args.addAll(con.getArgItems());
			}
		}
		return args;
	}
	
	/**
	 * Finds all textual units that are in the same coordination structure as the textual unit <var>conjunct</var>.
	 * 
	 * @param conjunct	a textual unit
	 * @return			all textual units coordinated with <var>conjunct</var>, or empty set if none
	 */
	public static LinkedHashSet<SurfaceElement> getConjuncts(SurfaceElement conjunct) {
		Document d = conjunct.getSentence().getDocument();
		LinkedHashSet<SurfaceElement> conjoined = new LinkedHashSet<>();
		LinkedHashSet<SemanticItem> conjs = Document.getSemanticItemsByClass(d, Conjunction.class);
		if (conjs.size() == 0) return conjoined;
		for (SemanticItem c: conjs) {
			Conjunction conj = (Conjunction)c;
			conjoined = getConjunctSurfaceElements(conj.getPredicate().getSurfaceElement());
			if (conjoined.contains(conjunct)) {
				conjoined.remove(conjunct);
				return conjoined;
			}
		}
		return new LinkedHashSet<SurfaceElement>();
	}
	
	/**
	 * Determines whether the textual unit <var>surf</var> is one that is conjoined by the coordinator
	 * <var>coord</var>.
	 * 
	 * @param coord	the coordinator
	 * @param surf	the potentially conjoined item
	 * 
	 * @return true if the textual unit is coordinated by the coordinator
	 */
	public static boolean isConjunctionArgument(SurfaceElement coord, SurfaceElement surf) {
		if (surf.hasSemantics() == false) return false;
		LinkedHashSet<SemanticItem> args = getConjunctionArguments(coord);
		if (args.size() == 0) return false;
		for (SemanticItem arg: args) {
			// Not sure why contains() did not work here. 
			for (SemanticItem c: surf.getSemantics())
				if (c.equals(arg)) return true;
		}
		return false;
	}
	
	/**
	 * Gets all textual units that are in a coordination structure indicated by <var>coord</var>.
	 * If the conjunct refers to a complex semantic item with a predicate, the textual unit corresponding
	 * to the predicate is used.
	 * 
	 * @param coord	the coordination indicator
	 * 
	 * @return	all textual units in the coordination structure
	 */
	public static LinkedHashSet<SurfaceElement> getConjunctSurfaceElements(SurfaceElement coord) {
		LinkedHashSet<SemanticItem> conjs = filterByConjunctions(coord);
		LinkedHashSet<SurfaceElement> args = new LinkedHashSet<>();
		if (conjs.size() == 0) return args;
		for (SemanticItem conj: conjs) {
			if (conj instanceof Conjunction) {
				Conjunction con = (Conjunction)conj;
				List<SemanticItem> conArgs = con.getArgItems();
				for (SemanticItem arg: conArgs) {
					if (arg instanceof AbstractTerm) {
						args.add(((AbstractTerm)arg).getSurfaceElement());
					} else if (arg instanceof HasPredicate) {
						args.add((((HasPredicate)arg).getPredicate().getSurfaceElement()));
					}
				}
			}
		}
		return args;
	}
	
	/**
	 * Gets all conjunction relations from the document that include a semantic item <var>sem</var> 
	 * as an argument or a predicate.
	 * 
	 * @param doc	the document with the semantic item
	 * @param sem	the semantic item, potentially in a conjunction relation
	 * 
	 * @return		the conjunction relation with the semantic item if any, null otherwise
	 */
	public static Conjunction getConjunctionsWithSemanticItem(Document doc, SemanticItem sem) {
		if (doc.getSemanticItems() == null) return null;
		LinkedHashSet<SemanticItem> conjs = Document.getSemanticItemsByClass(doc, Conjunction.class);
		if (conjs.size() == 0) return null;
		for (SemanticItem c: conjs) {
			Conjunction conj = (Conjunction)c;
			Predicate pred = conj.getPredicate();
			if ( (pred != null && conj.getPredicate().equals(sem)) || conj.getArgItems().contains(sem) )
				return conj;
		}
		return null;
	}
	
	/**
	 * Gets semantic items that are coordinated with or by the semantic item <var>sem</var>.
	 * 
	 * @param doc	the document with the semantic item
	 * @param sem	the semantic item, potentially in a conjunction relation or indicating one
	 * @return		semantic items linked to <var>sem</var> with a conjunction relation
	 */
	public static LinkedHashSet<SemanticItem> getConjunctSemanticItems(Document doc, SemanticItem sem) {
		Conjunction conj = getConjunctionsWithSemanticItem(doc,sem);
		if (conj == null) return null;
		LinkedHashSet<SemanticItem> outItems = new LinkedHashSet<SemanticItem>();
		for (SemanticItem c: conj.getArgItems()) {
			if (c.equals(sem)) continue;
			outItems.add(c);
		}
		return outItems;
	}
	
	
	/**
	 * Finds the coordinating conjunction that licenses the syntactic dependency 
	 * <var>synDep</var> in the sentence <var>sent</var>.<p>
	 * <var>synDep</var> is expected to be a dependency of type <i>conjunct</i>.
	 * If there are multiple possibilities, returns the one that is closest to the
	 * end of the sentence. 
	 * 
	 * @param sent		the sentence
	 * @param synDep	the	conjunct syntactic dependency
	 * @return			the coordinator if any, null otherwise
	 */
	public static SurfaceElement conjSurfaceElement(Sentence sent, SynDependency synDep) {
		if (synDep.getType().startsWith("conj_") == false) return null;
		SurfaceElement gov = synDep.getGovernor();
		SurfaceElement dep = synDep.getDependent();
		SurfaceElement left = (SpanList.atLeft(gov.getSpan(),dep.getSpan()) ? gov : dep);
		SurfaceElement right = (SpanList.atLeft(gov.getSpan(),dep.getSpan()) ? dep : gov);
		List<SurfaceElement> sItems = sent.getSurfaceElementsFromSpan(new Span(left.getSpan().getEnd(),right.getSpan().getBegin()));
		List<SurfaceElement> possibleConjs = new ArrayList<>();
		if (sItems == null) return null;
		String sub = synDep.getType().substring(synDep.getType().indexOf("_")+1);
		for (SurfaceElement si: sItems) {
			if (si.getText().equals(sub)) {
				possibleConjs.add(si);
			}
		}
		if (possibleConjs.size() == 0) {
			for (SurfaceElement si: sItems) {
				log.log(Level.FINEST,"Coordinating conjunction for {0}_{1}: {2}.", new Object[]{synDep.toShortString(),sub,si.getText()});
				if ((sub.equals("and") && AND_COORDINATORS.contains(si.getText())) || 
					// weird that this happens, but it happens PMID 8617979
					(sub.equals("plus") && AND_COORDINATORS.contains(si.getText())) ||
					(sub.equals("or") && OR_COORDINATORS.contains(si.getText())) || 
					(sub.equals("negcc") && NEG_COORDINATORS.contains(si.getText())))
					possibleConjs.add(si);
			}
		}
		if (possibleConjs.size() == 0) return null;
		if (possibleConjs.size() == 1) return possibleConjs.get(0);
		// not very good
		// find all candidates, and keep removing them 
		// until you find a good one
		return possibleConjs.get(possibleConjs.size()-1);
	}
	
	/**
	 * Finds the semantic types of the conjunction relation.<p>
	 * If the conjuncts are relations or predicates, the semantic type is simply <i>PRED</i>.
	 * Otherwise, it takes the semantic types common to the conjuncts as the semantic type.
	 * 
	 * @param conj	the conjunction relation	
	 * @return		the set of common semantic types, or <i>PRED</i> if the arguments are relations or predicates
	 */
	public static Set<String> conjunctionSemanticTypes(Conjunction conj) {
		List<SemanticItem> args = conj.getArgItems();
		Set<String> types = null;
		for (int i =0; i < args.size(); i++) {
			SemanticItem a = args.get(i);
			Set<String> semtypes = new HashSet<>();
			if (a instanceof Predicate || a instanceof Relation) semtypes.add("PRED");
			else semtypes = a.getAllSemtypes();
			if (types == null)  types = semtypes;
			else types.retainAll(semtypes);
		}
		return types;
	}
	
	/**
	 * Similar to {@code #conjunctionSemanticTypes(Conjunction)}, but checks semantic groups instead of types.
	 * 
	 * @param conj	the conjunction relation
	 * @return		the set of common semantic groups, or PRED
	 */
	public static Set<String> conjunctionSemanticGroups(Conjunction conj) {
		List<SemanticItem> args = conj.getArgItems();
		Set<String> groups = null;
		for (int i =0; i < args.size(); i++) {
			SemanticItem a = args.get(i);
			Set<String> semgroups = new HashSet<>();
			if (a instanceof Predicate || a instanceof Relation) semgroups.add("PRED");
			else semgroups = SemUtils.getSemGroups(a);
			if (groups == null)  groups = semgroups;
			else groups.retainAll(semgroups);
		}
		return groups;
	}
	
	/**
	 * Gets the common semantic type (or group, no common semantic type exists) between two semantic items.<p>
	 * <i>PRED</i> is returned, if one or both of the parameters are <code>Relation</code> or <code>Predicate</code> objects.
	 * 
	 * @param a	the first semantic item
	 * @param b	the second semantic item	
	 * @return	the common semantic type/group for entities, null if no common type/group
	 */
	public static String semanticallyConsonant(SemanticItem a, SemanticItem b) {
		if (a instanceof Predicate && b instanceof Predicate) return "PRED";
		if (a instanceof Relation && b instanceof Relation) return "PRED";
		if (a.getType().equals(b.getType())) return a.getType();
		String type = null;
		if (a instanceof Entity && b instanceof Entity) {
			Entity ae = (Entity)a; 
			Entity be = (Entity)b;
			type = SemUtils.matchingSemType(ae, be, false);
			if (type == null) {
				type = SemUtils.matchingSemGroup(ae, be);
			}
		}
		return type;
	}
	
	/**
	 * Gets the common semantic type (or group, no common semantic type exists) between two textual units.<p>
	 * 
	 * @param a	the first textual unit
	 * @param b	the second textual unit	
	 * @return	the common semantic type/group for textual units, null if no common type/group
	 */
	public static String semanticallyConsonant(SurfaceElement a, SurfaceElement b) {
		LinkedHashSet<SemanticItem> as = a.getSemantics();
		LinkedHashSet<SemanticItem> bs = b.getSemantics();
		if (as == null || bs == null) return null;
		for (SemanticItem a1: as) {
			for (SemanticItem b1: bs) {
				String type = semanticallyConsonant(a1,b1);
				if (type != null) return type;
			}
		}
		return null;
	}
	
	/**
	 * Determines whether two textual units are coordinated, based on the dependency
	 * path between them. <p>
	 * They are taken to be coordinated if the path consists only of <i>conjunct</i> dependencies.
	 *   
	 * @param surf1	the first textual unit
	 * @param surf2	the second textual unit
	 * @return		true if only conjunct dependencies exist along the path 
	 */
	public static boolean conjunctsByDependencyPath(SurfaceElement surf1, SurfaceElement surf2) {
		if (DiscourseUtils.sameSentence(surf1, surf2) == false) 
			return false;
		List<SynDependency> sentDeps = surf1.getSentence().getEmbeddings();
		List<SynDependency> path = SynDependency.findDependencyPath(sentDeps, surf1, surf2, false);
		if (path == null) return false;
		return (SynDependency.dependenciesWithType(path, "conj", false).size() == path.size());
	}
	
	/**
	 * Determines potential coordination, based on the dependency path between the textual units and 
	 * whether they are adjacent. <p>
	 * It relies on {@code #conjunctsByDependencyPath(SurfaceElement, SurfaceElement)} and
	 * {@code #mayBeContiguous(SurfaceElement, SurfaceElement)}.
	 * 
	 * @param surf1	the first textual unit
	 * @param surf2	the second textual unit
	 * @return	true if the textual units are potential conjuncts
	 */
	public static boolean potentiallyCoordinated(SurfaceElement surf1, SurfaceElement surf2) {
		if (conjunctsByDependencyPath(surf1,surf2)) return false;
		if (DiscourseUtils.sameSentence(surf1, surf2) == false) 
			return false;
		// This gives good results, but not sure we should use it all the time
		// TODO Can be dangerous
		if (ConjunctionDetection.semanticallyConsonant(surf1, surf2) == null) return false;
		return mayBeContiguous(surf1,surf2);
	}
	
	/**
	 * Determines whether two textual units are contiguous. <p>
	 * Two textual units are contiguous if:<ul>
	 * <li> They are only separated by coordinating punctuations and conjunctions OR
	 * <li> If the intervening word has a conjunction dependency path with one of the textual units
	 * </ul>
	 * If there are intervening parentheses, the parenthesis content is ignored.
	 * 
	 * @param surf1	the first textual unit
	 * @param surf2	the second textual unit
	 * @return		true if the textual units are contiguous
	 */
	public static boolean mayBeContiguous(SurfaceElement surf1, SurfaceElement surf2) {
		SpanList intervene = null;
		if (SpanList.atLeft(surf1.getSpan(), surf2.getSpan())) {
			intervene = new SpanList(surf1.getSpan().getEnd(),surf2.getSpan().getBegin());
		}else {
			intervene = new SpanList(surf2.getSpan().getEnd(),surf1.getSpan().getBegin());
		}
		Sentence sent = surf1.getSentence();
		List<SurfaceElement> interveneSurfs = sent.getSurfaceElementsFromSpan(intervene);
		if (interveneSurfs.size() == 0) return false;
		if (interveneSurfs.size() == 1) {
			SurfaceElement between = interveneSurfs.get(0);
			if (between.containsAnyLemma(CONJ_SEPARATORS) || between.isCoordConj()) return true;
		}
		else if (interveneSurfs.size() == 2) {
			SurfaceElement first = interveneSurfs.get(0);
			SurfaceElement second = interveneSurfs.get(1);
			if (first.containsAnyLemma(CONJ_SEPARATORS) && second.isCoordConj()) return true;
			if (first.isCoordConj() && second.containsAnyLemma(CONJ_SEPARATORS)) return true;
		} else {
			int accountedFor = 0;
			for (SurfaceElement intv: interveneSurfs) {
				if (intv.containsAnyLemma(CONJ_SEPARATORS) || intv.isCoordConj() || 
						conjunctsByDependencyPath(surf1,intv))  {
					accountedFor++;
					continue;
				}
				break;
			}
			if (accountedFor == interveneSurfs.size()) return true;
		}
		return mayBeContiguousWithParentheses(interveneSurfs,CONJ_SEPARATORS);	
	}

	// checks whether the intervening tokens between two textual units consist of a complete parenthesis in addition to 
	// coordinating punctuations and conjunctions.
	private static boolean mayBeContiguousWithParentheses(List<SurfaceElement> intervening, List<String> separators) {
		if (intervening.get(0).containsToken("(") == false) return false;
		int rbs = 0;
		int lastRbs = 0;
		int rindex = -1;
		boolean breakWithRight = false;
		for (int i =0; i < intervening.size(); i++) {
			SurfaceElement surf = intervening.get(i);
			if (surf.containsToken("(")) rbs++;
			if (surf.containsToken(")")) {rindex = i; breakWithRight = true; break;}
		}
		if (!breakWithRight) return false;
		if (rindex > 0) {
			for (int i=rindex; i< intervening.size(); i++) {
				SurfaceElement surf = intervening.get(i);
				if (surf.containsToken("(")) { if (rbs > 0) break;}			
				if (surf.containsToken(")")) { rbs--; lastRbs = i;}
				else break;
			}
		}
		if (rbs > 0) return false;
		if (rbs == 0 ) {
			for (int i=intervening.size()-1; i>lastRbs; i--) {
				SurfaceElement surf = intervening.get(i);
				if (surf.containsAnyLemma(separators) == false && surf.isCoordConj() == false) return false;
			}
		}
		return true;
	}
	
	private static boolean spanWithConjElementsOnly(Document doc, Span sp) {
		List<SurfaceElement> intervening = doc.getSurfaceElementsInSpan(sp);
		for (SurfaceElement intv: intervening) {
			if (intv.containsAnyLemma(CONJ_SEPARATORS) == false || intv.isCoordConj() == false)  return false;
		}
		return true;
	}
}
